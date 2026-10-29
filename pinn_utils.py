"""Shared helpers for loading, evaluating, and logging fuel-cell PINN models.

Used by:
  - compare_pinn_models.py   (compare two trained checkpoints)
  - plot_pinn_model.py       (plot one trained checkpoint)
  - FuelCellPINN_*.py        (TrainingLog records loss history and training time)

Loading a model never trains it. The physics of each model (PDE residuals,
boundary conditions, geometry, loss weights) is read from that model's own
training script: only the definitions before section "# 5)" are executed.
To support a new model version, add its training script to MODEL_SCRIPTS.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Select the backend before importing DeepXDE.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import numpy as np
import torch


# Training script that defines each model version. The version is read from the
# checkpoint name, e.g. pinn_model_V2.2_<timestamp>-<step>.pt -> "V2.2".
MODEL_SCRIPTS = {
    "V1.2": "FuelCellPINN_V1,2_RAR.py",
    "V1.3": "FuelCellPINN_V1,3_RAR.py",
    "V1.3_exp": "FuelCellPINN_V1,3_RAR_training_display.py",
    "V2.2": "FuelCellPINN_V2,2.py",
    "V3": "FuelCellPINN_V3.py",
    "V3_exp": "FuelCellPINN_V3-exp.py",
}

# Trained models are saved in one subfolder per version, e.g. models/v3/
MODELS_DIRECTORY = Path("models")

# Matches the save format pinn_model_<version>_<YYYY-MM-DD>_<HHMM>-<step>.pt
# (older checkpoints without a version or timestamp also match)
CHECKPOINT_NAME = re.compile(
    r"^pinn_model(?:_(?P<version>.+?))?"
    r"(?:_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{4}))?"
    r"-(?P<step>\d+)$"
)

CHUNK_SIZE = 5000  # points per autograd pass, limits memory use


class PinnModelError(Exception):
    """A checkpoint or training script cannot be loaded."""


# ---------------------------------------------------------------------------
# Checkpoints and training scripts
# ---------------------------------------------------------------------------
def resolve_checkpoint(path: Path) -> Path | None:
    """Return the checkpoint file for `path` (with or without .pt), or None."""
    path = Path(path)
    candidates = [path, path.with_suffix(".pt")] if path.suffix != ".pt" else [path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def model_directory(version: str | None) -> Path:
    """Folder for one model version: "V3" -> models/v3/, no version -> models/unversioned/."""
    return MODELS_DIRECTORY / (version.lower() if version else "unversioned")


def available_checkpoints() -> list[Path]:
    """Checkpoints in models/ and its version subfolders, oldest first."""
    return sorted(MODELS_DIRECTORY.rglob("*.pt"), key=lambda p: p.stat().st_mtime)


def parse_checkpoint_name(path: Path) -> dict[str, Any]:
    match = CHECKPOINT_NAME.match(Path(path).stem)
    if not match:
        return {"version": None, "trained": None, "step": None}
    trained = None
    if match["date"]:
        trained = f"{match['date']} {match['time'][:2]}:{match['time'][2:]}"
    return {"version": match["version"], "trained": trained, "step": int(match["step"])}


def resolve_script(checkpoint: Path, override: str | Path | None = None) -> Path:
    """Find the training script that defines the physics of `checkpoint`."""
    if override:
        script = Path(override)
    else:
        version = parse_checkpoint_name(checkpoint)["version"]
        if version not in MODEL_SCRIPTS:
            raise PinnModelError(
                f"No training script is known for {Path(checkpoint).name} (version: {version}). "
                "Add the version to MODEL_SCRIPTS in pinn_utils.py or pass the script explicitly."
            )
        script = Path(MODEL_SCRIPTS[version])
    if not script.is_file():
        raise PinnModelError(f"Training script not found: {script}")
    return script


def parse_names(pattern: str, source: str) -> list[str] | None:
    match = re.search(pattern, source, re.S)
    if not match:
        return None
    return [name.strip() for name in match.group(1).split(",") if name.strip()]


def load_definitions(script: Path) -> dict[str, Any]:
    """Run only the definition sections (1-4) of a training script.

    Everything from "# 5)" onward (data object, network, training, saving,
    plotting) is skipped, so no model is built or trained.
    """
    source = Path(script).read_text(encoding="utf-8")
    marker = re.search(r"^# ?5\)", source, re.M)
    if not marker:
        raise PinnModelError(
            f"{script} has no '# 5)' section marker; cannot separate definitions from training.")

    namespace: dict[str, Any] = {"__name__": "pinn_definitions", "__file__": str(script)}
    # The scripts print GPU information at import; keep the caller's output clean
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source[:marker.start()], str(script), "exec"), namespace)

    for name in ("physLoss", "geom", "bcs"):
        if name not in namespace:
            raise PinnModelError(f"{script} does not define '{name}' before section 5.")

    # Name each boundary condition after the variable it was assigned to
    bc_variables = {id(v): k for k, v in namespace.items() if isinstance(v, dde.icbc.BC)}
    bc_names = [bc_variables.get(id(bc), f"bc_{i}") for i, bc in enumerate(namespace["bcs"])]

    weights_match = re.search(r"^weights\s*=\s*(\[[^\]]*\])", source, re.M)
    activation = re.search(r'^activation\s*=\s*"([^"]+)"', source, re.M)
    initializer = re.search(r'^initializer\s*=\s*"([^"]+)"', source, re.M)

    return {
        "script": Path(script),
        "physLoss": namespace["physLoss"],
        "geom": namespace["geom"],
        "bcs": namespace["bcs"],
        "bc_names": bc_names,
        "equation_names": parse_names(r"def physLoss.*?return\s*\[([^\]]*)\]", source),
        "output_names": parse_names(r"Network outputs\s*\[([^\]]*)\]", source),
        "weights": ast.literal_eval(weights_match.group(1)) if weights_match else None,
        "activation": activation.group(1) if activation else "tanh",
        "initializer": initializer.group(1) if initializer else "Glorot normal",
    }


def load_network(checkpoint: Path, device: torch.device,
                 definitions: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the FNN from the checkpoint's layer shapes and load its weights.

    Only the network weights are restored; the optimizer state is read for
    information but never used.
    """
    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    weights = saved["model_state_dict"] if "model_state_dict" in saved else saved

    weight_keys = sorted(
        (key for key in weights if re.fullmatch(r"linears\.\d+\.weight", key)),
        key=lambda key: int(key.split(".")[1]),
    )
    if not weight_keys:
        raise PinnModelError(f"{checkpoint} does not look like a DeepXDE FNN checkpoint.")
    layer_size = [weights[weight_keys[0]].shape[1]] + [weights[k].shape[0] for k in weight_keys]

    geom_dim = definitions["geom"].dim
    if layer_size[0] != geom_dim:
        raise PinnModelError(f"{checkpoint} has {layer_size[0]} inputs but {definitions['script']} "
                             f"defines a {geom_dim}D geometry. Check MODEL_SCRIPTS.")
    output_names = definitions["output_names"]
    if output_names is None or len(output_names) != layer_size[-1]:
        if output_names is not None:
            print(f"Note: {definitions['script']} lists outputs {output_names}, "
                  f"but the checkpoint has {layer_size[-1]}; using generic names.")
        definitions["output_names"] = [f"output_{i}" for i in range(layer_size[-1])]

    dtype = weights[weight_keys[0]].dtype
    net = dde.nn.FNN(layer_size, definitions["activation"], definitions["initializer"]).to(dtype)
    net.load_state_dict(weights)
    net.to(device).eval()

    lbfgs_iterations = None
    optimizer_state = saved.get("optimizer_state_dict", {}).get("state", {})
    if 0 in optimizer_state and "n_iter" in optimizer_state[0]:
        lbfgs_iterations = int(optimizer_state[0]["n_iter"])

    info = {
        "checkpoint": str(checkpoint),
        "script": str(definitions["script"]),
        **parse_checkpoint_name(checkpoint),
        "layer_size": layer_size,
        "parameters": int(sum(p.numel() for p in net.parameters())),
        "dtype": str(dtype).replace("torch.", ""),
        "lbfgs_iterations": lbfgs_iterations,
        "outputs": definitions["output_names"],
        "domain": [definitions["geom"].bbox[0].tolist(), definitions["geom"].bbox[1].tolist()],
    }
    return net, info


@dataclass
class LoadedModel:
    """A trained network together with the physics it was trained on."""
    net: torch.nn.Module
    definitions: dict[str, Any]
    info: dict[str, Any]
    training_record: dict[str, Any] | None


def load_model(checkpoint: Path, device: torch.device, script: str | Path | None = None) -> LoadedModel:
    """Load a checkpoint, its physics definitions, and its training record (if saved)."""
    resolved = resolve_checkpoint(checkpoint)
    if resolved is None:
        raise PinnModelError(f"Checkpoint not found: {checkpoint}")
    # Load definitions right before the network, since scripts may change global
    # DeepXDE settings such as the default float type
    definitions = load_definitions(resolve_script(resolved, script))
    net, info = load_network(resolved, device, definitions)
    return LoadedModel(net, definitions, info, load_training_record(resolved))


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Evaluation points
# ---------------------------------------------------------------------------
def evaluation_grid(geom, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular size x size grid over the domain: returns (Z, R, points)."""
    low, high = geom.bbox
    Z, R = np.meshgrid(np.linspace(low[0], high[0], size), np.linspace(low[1], high[1], size))
    return Z, R, np.column_stack((Z.ravel(), R.ravel()))


def validation_points(geom, n: int) -> np.ndarray:
    """Fixed random points just inside the domain (same domain -> same points)."""
    low, high = geom.bbox
    margin = 1e-4 * (high - low)
    return np.random.default_rng(20261014).uniform(low + margin, high - margin, size=(n, geom.dim))


def boundary_points(geom, n: int) -> np.ndarray:
    """n points along each face of the domain's bounding box.

    A fixed RNG gives every model identical points; corners are excluded
    because multiple boundary conditions meet there.
    """
    rng = np.random.default_rng(20261015)
    low, high = geom.bbox
    margin = 1e-4 * (high - low)
    faces = []
    for dim in range(geom.dim):
        for value in (low[dim], high[dim]):
            face = rng.uniform(low + margin, high - margin, size=(n, geom.dim))
            face[:, dim] = value
            faces.append(face)
    return np.vstack(faces)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def to_tensor(net: torch.nn.Module, points: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    parameter = next(net.parameters())
    x = torch.as_tensor(points, dtype=parameter.dtype, device=parameter.device)
    return x.clone().requires_grad_(requires_grad)


def predict(net: torch.nn.Module, points: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net(to_tensor(net, points)).cpu().numpy()


def to_numpy(values: Any, rows: int) -> np.ndarray:
    return torch.as_tensor(values).detach().cpu().numpy().reshape(rows, -1)


def pde_residuals(model: LoadedModel, points: np.ndarray) -> np.ndarray:
    """Evaluate the training script's physLoss; returns shape (points, equations)."""
    chunks = []
    for start in range(0, len(points), CHUNK_SIZE):
        x = to_tensor(model.net, points[start:start + CHUNK_SIZE], requires_grad=True)
        residuals = model.definitions["physLoss"](x, model.net(x))
        if not isinstance(residuals, (list, tuple)):
            residuals = [residuals]
        chunks.append(np.hstack([to_numpy(r, len(x)) for r in residuals]))
        dde.grad.clear()
    return np.vstack(chunks)


def equation_names(model: LoadedModel, residuals: np.ndarray) -> list[str]:
    """Names of the PDE residuals, falling back to generic names if they don't match."""
    names = model.definitions["equation_names"]
    if names is None or len(names) != residuals.shape[1]:
        return [f"pde_{i}" for i in range(residuals.shape[1])]
    return names


def error_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mse": float(np.mean(values**2)),
        "mae": float(np.mean(np.abs(values))),
        "max_abs": float(np.max(np.abs(values))),
        "points": int(values.size),
    }


def boundary_errors(model: LoadedModel, points: np.ndarray) -> dict[str, dict[str, float] | None]:
    """Evaluate each boundary condition exactly as DeepXDE does during training."""
    results = {}
    # DeepXDE caches boundary values by array id, so keep every filtered array alive
    kept = []
    for name, bc in zip(model.definitions["bc_names"], model.definitions["bcs"]):
        X = bc.filter(points)
        kept.append(X)
        if len(X) == 0:
            results[name] = None
            continue
        x = to_tensor(model.net, X, requires_grad=True)
        error = bc.error(X, x, model.net(x), 0, len(X))
        results[name] = error_metrics(to_numpy(error, len(X)))
        dde.grad.clear()
    return results


def training_objective(model: LoadedModel, pde_metrics: dict, bc_metrics: dict) -> float | None:
    """The model's own weighted training loss (sum of weighted MSE terms), or None."""
    terms = [m["mse"] for m in pde_metrics.values()] + [
        m["mse"] if m else None for m in bc_metrics.values()]
    weights = model.definitions["weights"]
    if weights is None or len(weights) != len(terms) or None in terms:
        return None
    return float(sum(w * t for w, t in zip(weights, terms)))


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward_flops(layer_size: list[int]) -> int:
    """Multiply-adds of the linear layers per input point (activations excluded)."""
    return int(sum(2 * a * b for a, b in zip(layer_size[:-1], layer_size[1:])))


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


# ---------------------------------------------------------------------------
# Training record (loss history and training time)
# ---------------------------------------------------------------------------
def training_record_path(checkpoint: Path) -> Path:
    """models/v3/pinn_model_V3_<ts>-<step>.pt -> models/v3/pinn_model_V3_<ts>-<step>_training.json"""
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(f"{checkpoint.stem}_training.json")


def load_training_record(checkpoint: Path) -> dict[str, Any] | None:
    path = training_record_path(checkpoint)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _to_list(values) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]


class TrainingLog:
    """Records loss history, training time, and RAR points for one training run.

    Usage in a training script:

        training_log = TrainingLog(model)
        with training_log.stage("Adam (initial)"):
            model.train(iterations=1000)
        training_log.add_rar_points(new_points, rar_round=1)
        ...
        save_path = model.save(...)
        training_log.save(save_path)

    The record is written next to the checkpoint as <checkpoint>_training.json
    and is plotted by plot_pinn_model.py.
    """

    def __init__(self, model: dde.Model):
        self.model = model
        self.started = datetime.now()
        self.start_time = time.perf_counter()
        self.stages: list[dict[str, Any]] = []
        self.rar_points: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def stage(self, name: str):
        """Time one model.train() call and remember its optimizer and loss weights."""
        start_step = self.model.train_state.step
        start = time.perf_counter()
        yield
        weights = self.model.loss_weights
        self.stages.append({
            "name": name,
            "optimizer": self.model.opt_name,
            # Losses in the history are multiplied by these weights (None = unweighted)
            "loss_weights": None if weights is None else _to_list(weights),
            "start_step": int(start_step),
            "end_step": int(self.model.train_state.step),
            "seconds": time.perf_counter() - start,
        })

    def add_rar_points(self, points: np.ndarray, rar_round: int) -> None:
        self.rar_points.append({"round": rar_round, "points": np.asarray(points).tolist()})

    def save(self, checkpoint_path: str | Path) -> Path:
        history = self.model.losshistory
        record = {
            "checkpoint": Path(checkpoint_path).name,
            "started": f"{self.started:%Y-%m-%d %H:%M:%S}",
            "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            # Total includes work between stages (e.g. RAR residual evaluation)
            "total_seconds": time.perf_counter() - self.start_time,
            "training_seconds": sum(stage["seconds"] for stage in self.stages),
            "device": str(default_device()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "stages": self.stages,
            "loss_history": {
                "steps": [int(s) for s in history.steps],
                "loss_train": [_to_list(loss) for loss in history.loss_train],
                "loss_test": [_to_list(loss) for loss in history.loss_test],
            },
            "rar_points": self.rar_points,
        }
        path = training_record_path(checkpoint_path)
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"Training record saved: {path} "
              f"(training time {record['training_seconds'] / 60:.1f} min)")
        return path
