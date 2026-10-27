"""Compare two pretrained DeepXDE/PyTorch fuel-cell PINNs (V1.x pipe-flow models).

Edit the configuration variables below, then run this file normally in VS Code.

This script never trains a model. Both checkpoints must already exist in the
models/ folder; if either is missing (or both point to the same file) the script
lists the available checkpoints and exits.

For each model it reports PDE residuals, boundary-condition errors, and inference
speed, then compares the two models' predicted fields directly.
"""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Select the backend before importing DeepXDE.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm




# ---------------------------------------------------------------------------
# User configuration: edit these before running the script.
# Paths are relative to the project root (the working directory in VS Code).
# ---------------------------------------------------------------------------
MODEL_A_PATH = Path("models/pinn_model_V1.3-12689.pt")
MODEL_B_PATH = Path("models/pinn_model_V1.3-12661.pt")
MODEL_A_LABEL = "original"
MODEL_B_LABEL = "experimental"




# Each run gets its own folder: metrics/model_comparisons/<date_time>_<A>_vs_<B>/
OUTPUT_DIRECTORY = Path("metrics/model_comparisons")
VALIDATION_POINTS = 10000
BOUNDARY_SAMPLES = 1000
GRID_SIZE = 200
INFERENCE_REPEATS = 15

# ---------------------------------------------------------------------------
# Model definition: keep these in sync with the training code (V1.x).
# The layer sizes are read from each checkpoint, so they do not need to be set.
# ---------------------------------------------------------------------------
ACTIVATION = "tanh"
INITIALIZER = "Glorot normal"

RHO = 998.0
MU = 0.001002
U0 = 0.01
L = 0.1
R = 0.0075
L_R = L / R
RE = RHO * U0 * 2.0 * R / MU

OUTPUT_NAMES = ["u_z", "u_r", "p"]
PDE_NAMES = ["axial_momentum", "radial_momentum", "continuity"]

# Matches the save format pinn_model_<version>_<YYYY-MM-DD>_<HHMM>-<step>.pt
# (older checkpoints without a version or timestamp also match)
CHECKPOINT_NAME = re.compile(
    r"^pinn_model(?:_(?P<version>.+?))?"
    r"(?:_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{4}))?"
    r"-(?P<step>\d+)$"
)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def resolve_checkpoint(path: Path) -> Path | None:
    """Return the checkpoint file for `path` (with or without .pt), or None."""
    candidates = [path, path.with_suffix(".pt")] if path.suffix != ".pt" else [path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def require_two_checkpoints() -> tuple[Path, Path]:
    """Exit unless both configured checkpoints exist and are different files."""
    checkpoint_a = resolve_checkpoint(MODEL_A_PATH)
    checkpoint_b = resolve_checkpoint(MODEL_B_PATH)

    problems = []
    if checkpoint_a is None:
        problems.append(f"Model A checkpoint not found: {MODEL_A_PATH}")
    if checkpoint_b is None:
        problems.append(f"Model B checkpoint not found: {MODEL_B_PATH}")
    if checkpoint_a and checkpoint_b and checkpoint_a.resolve() == checkpoint_b.resolve():
        problems.append(f"Model A and Model B are the same file: {checkpoint_a}")

    if problems:
        print("Cannot compare models:")
        for problem in problems:
            print(f"  - {problem}")
        available = sorted(Path("models").glob("*.pt"), key=lambda p: p.stat().st_mtime)
        print("\nAvailable checkpoints in models/ (oldest first):")
        for path in available:
            print(f"  {path}")
        if not available:
            print("  (none)")
        print("\nSet MODEL_A_PATH and MODEL_B_PATH to two different checkpoints. "
              "This script does not train models.")
        sys.exit(1)

    return checkpoint_a, checkpoint_b


def parse_checkpoint_name(path: Path) -> dict[str, Any]:
    match = CHECKPOINT_NAME.match(path.stem)
    if not match:
        return {"version": None, "trained": None, "step": None}
    trained = None
    if match["date"]:
        trained = f"{match['date']} {match['time'][:2]}:{match['time'][2:]}"
    return {"version": match["version"], "trained": trained, "step": int(match["step"])}


def load_network(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
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
        sys.exit(f"{checkpoint} does not look like a DeepXDE FNN checkpoint.")
    layer_size = [weights[weight_keys[0]].shape[1]] + [weights[k].shape[0] for k in weight_keys]

    if layer_size[0] != 2 or layer_size[-1] != len(OUTPUT_NAMES):
        sys.exit(
            f"{checkpoint} has inputs/outputs {layer_size[0]}/{layer_size[-1]}, but this "
            "script compares V1.x pipe-flow models (inputs z, r; outputs u_z, u_r, p)."
        )

    dtype = weights[weight_keys[0]].dtype
    net = dde.nn.FNN(layer_size, ACTIVATION, INITIALIZER).to(dtype)
    net.load_state_dict(weights)
    net.to(device).eval()

    lbfgs_iterations = None
    optimizer_state = saved.get("optimizer_state_dict", {}).get("state", {})
    if 0 in optimizer_state and "n_iter" in optimizer_state[0]:
        lbfgs_iterations = int(optimizer_state[0]["n_iter"])

    info = {
        "checkpoint": str(checkpoint),
        **parse_checkpoint_name(checkpoint),
        "layer_size": layer_size,
        "parameters": int(sum(p.numel() for p in net.parameters())),
        "dtype": str(dtype).replace("torch.", ""),
        "lbfgs_iterations": lbfgs_iterations,
    }
    return net, info


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


def gradient(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(output.sum(), x, create_graph=True)[0]


def pde_residuals(net: torch.nn.Module, points: np.ndarray) -> np.ndarray:
    """Axial momentum, radial momentum, and continuity residuals (same as physLoss)."""
    x = to_tensor(net, points, requires_grad=True)
    y = net(x)
    u_z, u_r = y[:, 0], y[:, 1]
    r = x[:, 1]

    du_z = gradient(u_z, x)
    du_r = gradient(u_r, x)
    dp = gradient(y[:, 2], x)
    du_z_z, du_z_r = du_z[:, 0], du_z[:, 1]
    du_r_z, du_r_r = du_r[:, 0], du_r[:, 1]
    dp_z, dp_r = dp[:, 0], dp[:, 1]

    du_z_zz = gradient(du_z_z, x)[:, 0]
    du_z_rr = gradient(du_z_r, x)[:, 1]
    du_r_zz = gradient(du_r_z, x)[:, 0]
    du_r_rr = gradient(du_r_r, x)[:, 1]

    nu_star = 2.0 / RE
    loss_z_mom = r * (u_r * du_z_r + u_z * du_z_z + dp_z) - nu_star * (
        r * (du_z_rr + du_z_zz) + du_z_r
    )
    loss_r_mom = r**2 * (u_r * du_r_r + u_z * du_r_z + dp_r) - nu_star * (
        r**2 * (du_r_rr + du_r_zz) + r * du_r_r - u_r
    )
    loss_cont = r * (du_r_r + du_z_z) + u_r
    return torch.stack((loss_z_mom, loss_r_mom, loss_cont), dim=1).detach().cpu().numpy()


def error_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.mean(np.abs(values))),
        "max_abs": float(np.max(np.abs(values))),
    }


def boundary_points(n: int) -> dict[str, np.ndarray]:
    # Fixed RNG so both models see identical points; corners are excluded
    # because multiple boundary conditions meet there.
    rng = np.random.default_rng(20261015)
    z = rng.uniform(1e-4, L_R - 1e-4, n)
    r = rng.uniform(1e-4, 1.0 - 1e-4, n)
    return {
        "top": np.column_stack((z, np.ones(n))),
        "axis": np.column_stack((z, np.zeros(n))),
        "inlet": np.column_stack((np.zeros(n), r)),
        "outlet": np.column_stack((np.full(n, L_R), r)),
    }


def boundary_errors(net: torch.nn.Module, points: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    top = predict(net, points["top"])
    axis = predict(net, points["axis"])
    inlet = predict(net, points["inlet"])
    outlet = predict(net, points["outlet"])

    x = to_tensor(net, points["axis"], requires_grad=True)
    y = net(x)
    du_z_dr = gradient(y[:, 0], x)[:, 1].detach().cpu().numpy()
    dp_dr = gradient(y[:, 2], x)[:, 1].detach().cpu().numpy()

    return {
        "top_wall_u_z_equals_0": error_metrics(top[:, 0]),
        "top_wall_u_r_equals_0": error_metrics(top[:, 1]),
        "axis_u_r_equals_0": error_metrics(axis[:, 1]),
        "axis_du_z_dr_equals_0": error_metrics(du_z_dr),
        "axis_dp_dr_equals_0": error_metrics(dp_dr),
        "inlet_u_z_equals_1": error_metrics(inlet[:, 0] - 1.0),
        "inlet_u_r_equals_0": error_metrics(inlet[:, 1]),
        "outlet_p_equals_0": error_metrics(outlet[:, 2]),
    }


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference(net: torch.nn.Module, points: np.ndarray, repeats: int) -> list[dict[str, Any]]:
    device = next(net.parameters()).device
    results = []
    for size in sorted({1, 100, 1000, 10000, len(points)}):
        batch = points[:min(size, len(points))]
        # Warm-up avoids counting first-call setup overhead in the timed samples
        for _ in range(2):
            predict(net, batch)
        timings_ms = []
        for _ in range(repeats):
            sync_device(device)
            start = time.perf_counter()
            predict(net, batch)
            sync_device(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
        median_ms = float(statistics.median(timings_ms))
        results.append({
            "batch_size": len(batch),
            "median_ms": median_ms,
            "p95_ms": float(np.percentile(timings_ms, 95)),
            "points_per_second": float(len(batch) / (median_ms / 1000.0)) if median_ms > 0 else None,
        })
    return results


def evaluate_model(label: str, checkpoint: Path, device: torch.device,
                   points: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    print(f"\n{'=' * 72}\nEvaluating {label}: {checkpoint}\n{'=' * 72}")
    net, info = load_network(checkpoint, device)

    grid_predictions = predict(net, points["grid"])
    validation_predictions = predict(net, points["validation"])
    validation_residuals = pde_residuals(net, points["validation"])
    grid_residuals = pde_residuals(net, points["grid"])

    result = {
        "label": label,
        "model": info,
        "pde_residuals": {
            name: error_metrics(validation_residuals[:, i]) for i, name in enumerate(PDE_NAMES)
        },
        "boundary_errors": boundary_errors(net, points["boundary"]),
        "inference_timing": benchmark_inference(net, points["grid"], INFERENCE_REPEATS),
        "predictions_finite": bool(np.isfinite(grid_predictions).all()
                                   and np.isfinite(validation_predictions).all()),
    }

    print(f"Version: {info['version']} | Trained: {info['trained']} | Step: {info['step']} "
          f"| L-BFGS iterations: {info['lbfgs_iterations']}")
    print(f"Layers: {info['layer_size']} | Parameters: {info['parameters']:,} | dtype: {info['dtype']}")
    if not result["predictions_finite"]:
        print("WARNING: predictions contain NaN or infinity.")

    arrays = {
        "grid_predictions": grid_predictions,
        "validation_predictions": validation_predictions,
        "validation_residuals": validation_residuals,
        "grid_residuals": grid_residuals,
    }
    return result, arrays


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def prediction_difference(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    output = {}
    for i, name in enumerate(OUTPUT_NAMES):
        difference = b[:, i] - a[:, i]
        denominator = float(np.linalg.norm(a[:, i]))
        output[name] = {
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "max_abs": float(np.max(np.abs(difference))),
            "relative_l2": float(np.linalg.norm(difference) / denominator) if denominator > 1e-12 else None,
        }
    return output


def summary_rows(result_a: dict[str, Any], result_b: dict[str, Any]) -> list[dict[str, Any]]:
    """Scalar metrics side by side, where lower is better for every row."""
    rows = []
    for name in PDE_NAMES:
        rows.append((f"PDE {name} RMSE", result_a["pde_residuals"][name]["rmse"],
                     result_b["pde_residuals"][name]["rmse"]))
    for name in result_a["boundary_errors"]:
        rows.append((f"BC {name} RMSE", result_a["boundary_errors"][name]["rmse"],
                     result_b["boundary_errors"][name]["rmse"]))
    rows.append((f"Inference ms (n={result_a['inference_timing'][-1]['batch_size']:,})",
                 result_a["inference_timing"][-1]["median_ms"],
                 result_b["inference_timing"][-1]["median_ms"]))

    output = []
    for metric, a, b in rows:
        change = (b - a) / a * 100.0 if a else None
        better = "tie" if np.isclose(a, b, rtol=1e-3, atol=0) else ("A" if a < b else "B")
        output.append({"metric": metric, "a": a, "b": b, "change_percent": change, "better": better})
    return output


def print_summary(rows: list[dict[str, Any]], differences: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}\nSummary: A = {MODEL_A_LABEL}, B = {MODEL_B_LABEL} (lower is better)\n{'=' * 72}")
    print(f"{'Metric':<38} {'A':>11} {'B':>11} {'B vs A':>9}  Better")
    for row in rows:
        change = f"{row['change_percent']:+.1f}%" if row["change_percent"] is not None else "--"
        print(f"{row['metric']:<38} {row['a']:>11.3e} {row['b']:>11.3e} {change:>9}  {row['better']}")

    wins_a = sum(row["better"] == "A" for row in rows)
    wins_b = sum(row["better"] == "B" for row in rows)
    print(f"\nLower on {wins_a} metrics: A | Lower on {wins_b} metrics: B")

    print("\nPrediction differences on validation points (B minus A):")
    for name, values in differences.items():
        relative = f"{values['relative_l2']:.3%}" if values["relative_l2"] is not None else "--"
        print(f"  {name:<4} RMSE={values['rmse']:.3e}  max={values['max_abs']:.3e}  relative L2={relative}")


def plot_metrics(result_a: dict[str, Any], result_b: dict[str, Any], plots_dir: Path) -> Path:
    """Bar charts of PDE residuals, boundary errors, and inference latency."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"PINN Comparison: {MODEL_A_LABEL} vs {MODEL_B_LABEL}", fontsize=14)
    width = 0.36

    def grouped_bars(ax, names, labels, key, title):
        x = np.arange(len(names))
        values_a = [max(result_a[key][n]["rmse"], 1e-16) for n in names]
        values_b = [max(result_b[key][n]["rmse"], 1e-16) for n in names]
        ax.bar(x - width / 2, values_a, width, label=MODEL_A_LABEL)
        ax.bar(x + width / 2, values_b, width, label=MODEL_B_LABEL)
        ax.set_title(title)
        ax.set_ylabel("RMSE (log scale), lower is better")
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    grouped_bars(axes[0], PDE_NAMES, [n.replace("_", " ") for n in PDE_NAMES],
                 "pde_residuals", "PDE Residual")
    boundary_names = list(result_a["boundary_errors"])
    grouped_bars(axes[1], boundary_names,
                 [n.replace("_equals_", " = ").replace("_", " ") for n in boundary_names],
                 "boundary_errors", "Boundary-Condition Error")

    for result, label in ((result_a, MODEL_A_LABEL), (result_b, MODEL_B_LABEL)):
        timing = result["inference_timing"]
        axes[2].plot([t["batch_size"] for t in timing], [t["median_ms"] for t in timing],
                     marker="o", label=label)
    axes[2].set_title("Inference Latency")
    axes[2].set_xlabel("Batch size (points)")
    axes[2].set_ylabel("Median time (ms), lower is better")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = plots_dir / "model_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_field_differences(Z: np.ndarray, RR: np.ndarray, grid_a: np.ndarray,
                           grid_b: np.ndarray, plots_dir: Path) -> Path:
    """Contours of where the two models' predicted fields disagree (B minus A)."""
    fig, axes = plt.subplots(len(OUTPUT_NAMES), 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Prediction difference: {MODEL_B_LABEL} minus {MODEL_A_LABEL}")
    for i, (ax, name) in enumerate(zip(axes, OUTPUT_NAMES)):
        difference = (grid_b[:, i] - grid_a[:, i]).reshape(Z.shape)
        # 99.5th percentile keeps a single inlet-corner spike from washing out the colors
        limit = max(float(np.percentile(np.abs(difference), 99.5)), 1e-12)
        contour = ax.contourf(Z, RR, np.clip(difference, -limit, limit),
                              levels=np.linspace(-limit, limit, 101), cmap="RdBu_r", extend="both")
        fig.colorbar(contour, ax=ax, label=f"Δ{name} (dimensionless)")
        ax.set_ylabel("r / R")
    axes[-1].set_xlabel("z / R")
    fig.tight_layout()
    path = plots_dir / "field_differences.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residual_maps(Z: np.ndarray, RR: np.ndarray, residuals_a: np.ndarray,
                       residuals_b: np.ndarray, plots_dir: Path) -> Path:
    """Local PDE residual of each model on a shared color scale."""
    local_a = np.maximum(np.sum(residuals_a.astype(np.float64)**2, axis=1), 1e-20).reshape(Z.shape)
    local_b = np.maximum(np.sum(residuals_b.astype(np.float64)**2, axis=1), 1e-20).reshape(Z.shape)
    norm = LogNorm(vmin=min(local_a.min(), local_b.min()), vmax=max(local_a.max(), local_b.max()))

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
    for ax, local, label in ((axes[0], local_a, MODEL_A_LABEL), (axes[1], local_b, MODEL_B_LABEL)):
        mesh = ax.pcolormesh(Z, RR, local, shading="auto", cmap="inferno", norm=norm)
        ax.set_title(f"Local PDE residual: {label}")
        ax.set_ylabel("r / R")
    axes[-1].set_xlabel("z / R")
    fig.colorbar(mesh, ax=axes, label="Sum of squared PDE residuals")
    path = plots_dir / "residual_maps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def write_run_summary(path: Path, run_time: datetime, checkpoint_a: Path, checkpoint_b: Path) -> None:
    """Small note recording which checkpoints were compared and when."""
    path.write_text(
        "# Model Comparison\n\n"
        f"- Run: {run_time:%Y-%m-%d %H:%M:%S}\n"
        f"- Model A ({MODEL_A_LABEL}): `{checkpoint_a.name}`\n"
        f"- Model B ({MODEL_B_LABEL}): `{checkpoint_b.name}`\n",
        encoding="utf-8",
    )


def main() -> None:
    # Check checkpoints before doing any other work
    checkpoint_a, checkpoint_b = require_two_checkpoints()

    run_time = datetime.now()
    output_dir = OUTPUT_DIRECTORY / (
        f"{safe_name(MODEL_A_LABEL)}_vs_{safe_name(MODEL_B_LABEL)}_{run_time:%Y-%m-%d_%H%M%S}"
    )
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Identical deterministic evaluation points for both models
    Z, RR = np.meshgrid(np.linspace(0.0, L_R, GRID_SIZE), np.linspace(0.0, 1.0, GRID_SIZE))
    rng = np.random.default_rng(20261014)
    points = {
        "grid": np.column_stack((Z.ravel(), RR.ravel())),
        "validation": np.column_stack((
            rng.uniform(0.0, L_R, VALIDATION_POINTS),
            rng.uniform(1e-4, 1.0 - 1e-4, VALIDATION_POINTS),
        )),
        "boundary": boundary_points(BOUNDARY_SAMPLES),
    }

    print("PINN pretrained-model comparison (no training is performed)")
    print(f"Domain: z/R in [0, {L_R:.4f}], r/R in [0, 1]")
    print(f"Validation points: {VALIDATION_POINTS:,} | Grid points: {len(points['grid']):,}")
    print(f"PyTorch {torch.__version__} | Device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    result_a, arrays_a = evaluate_model(MODEL_A_LABEL, checkpoint_a, device, points)
    result_b, arrays_b = evaluate_model(MODEL_B_LABEL, checkpoint_b, device, points)

    if result_a["model"]["layer_size"] != result_b["model"]["layer_size"]:
        print("\nNote: the two models have different architectures.")

    rows = summary_rows(result_a, result_b)
    differences = prediction_difference(arrays_a["validation_predictions"],
                                        arrays_b["validation_predictions"])
    print_summary(rows, differences)

    # Save metrics, predictions, and plots
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "deepxde": getattr(dde, "__version__", "unknown"),
        "device": str(device),
    }
    for result, arrays in ((result_a, arrays_a), (result_b, arrays_b)):
        safe_label = safe_name(result["label"])
        (output_dir / f"{safe_label}_metrics.json").write_text(
            json.dumps({**result, "runtime": runtime}, indent=2), encoding="utf-8")
        np.savez_compressed(output_dir / f"{safe_label}_predictions.npz",
                            grid_points=points["grid"], validation_points=points["validation"], **arrays)

    comparison = {
        "model_a": result_a["model"],
        "model_b": result_b["model"],
        "summary_lower_is_better": rows,
        "validation_prediction_differences_b_minus_a": differences,
        "grid_prediction_differences_b_minus_a": prediction_difference(
            arrays_a["grid_predictions"], arrays_b["grid_predictions"]),
        "runtime": runtime,
        "notes": [
            "Lower PDE residuals mean better consistency with the encoded equations, not necessarily physical accuracy.",
            "Boundary metrics are errors against the boundary conditions encoded in this script.",
            "Inference timing depends on hardware and system load.",
        ],
    }
    comparison_path = output_dir / "model_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    plot_paths = [
        plot_metrics(result_a, result_b, plots_dir),
        plot_field_differences(Z, RR, arrays_a["grid_predictions"], arrays_b["grid_predictions"], plots_dir),
        plot_residual_maps(Z, RR, arrays_a["grid_residuals"], arrays_b["grid_residuals"], plots_dir),
    ]

    write_run_summary(output_dir / "README.md", run_time, checkpoint_a, checkpoint_b)

    print(f"\nComparison saved to: {output_dir.resolve()}")
    for path in plot_paths:
        print(f"Plot saved to: {path.resolve()}")


if __name__ == "__main__":
    main()
