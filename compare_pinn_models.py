"""Benchmark and compare two pretrained DeepXDE/PyTorch fuel-cell PINNs.

Edit the configuration variables below, then run this file normally in VS Code.

This script does not train or modify either checkpoint. It recreates the network
and physics definition from the project script, restores each checkpoint, then
measures inference latency, PDE residuals, boundary-condition errors, and the
difference between the two models' predictions.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

# Select the backend before importing DeepXDE.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model definition: keep these values/functions in sync with the training code.
# ---------------------------------------------------------------------------
SEED = 42
RHO = 998.0
MU = 0.001002
U0 = 0.01
L = 0.1
R = 0.0075
L_R = L / R
RE = RHO * U0 * 2.0 * R / MU

# ---------------------------------------------------------------------------
# User configuration: edit these paths/settings before running the script.
# Paths are relative to the project root (the working directory in VS Code).
# ---------------------------------------------------------------------------
ORIGINAL_MODEL_PATH = Path("models/pinn_model_V1.3-12689.pt")
EXPERIMENTAL_MODEL_PATH = Path("models/pinn_model_V1.3-12661.pt")
OUTPUT_DIRECTORY = Path("metrics")
INFERENCE_REPEATS = 15
BOUNDARY_SAMPLES = 1000

BOUNDARY_PTS = 1000
INTERIOR_PTS = 5000
LOSS_WEIGHTS = [10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1]
LAYER_SIZE = [2] + [30] * 6 + [3]
ACTIVATION = "tanh"
INITIALIZER = "Glorot normal"


def physLoss(x, y):
    """Return axial momentum, radial momentum, and continuity residuals."""
    r = x[:, 1:2]
    u_z = y[:, 0:1]
    u_r = y[:, 1:2]

    du_z_z = dde.grad.jacobian(y, x, i=0, j=0)
    du_z_r = dde.grad.jacobian(y, x, i=0, j=1)
    du_r_z = dde.grad.jacobian(y, x, i=1, j=0)
    du_r_r = dde.grad.jacobian(y, x, i=1, j=1)
    dp_z = dde.grad.jacobian(y, x, i=2, j=0)
    dp_r = dde.grad.jacobian(y, x, i=2, j=1)

    du_z_zz = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_z_rr = dde.grad.hessian(y, x, component=0, i=1, j=1)
    du_r_zz = dde.grad.hessian(y, x, component=1, i=0, j=0)
    du_r_rr = dde.grad.hessian(y, x, component=1, i=1, j=1)

    nu_star = 2.0 / RE
    loss_z_mom = r * (u_r * du_z_r + u_z * du_z_z + dp_z) - nu_star * (
        r * (du_z_rr + du_z_zz) + du_z_r
    )
    loss_r_mom = r**2 * (u_r * du_r_r + u_z * du_r_z + dp_r) - nu_star * (
        r**2 * (du_r_rr + du_r_zz) + r * du_r_r - u_r
    )
    loss_cont = r * (du_r_r + du_z_z) + u_r
    return [loss_z_mom, loss_r_mom, loss_cont]


def top_wall(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1.0)


def bottom_wall(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)


def left_wall(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)


def right_wall(x, on_boundary):
    return on_boundary and np.isclose(x[0], L_R)


def build_model():
    """Rebuild the architecture/data needed to restore a saved checkpoint."""
    dde.config.set_random_seed(SEED)
    geom = dde.geometry.Rectangle([0.0, 0.0], [L_R, 1.0])

    bcs = [
        dde.icbc.DirichletBC(geom, lambda x: 0, top_wall, component=0),
        dde.icbc.DirichletBC(geom, lambda x: 0, top_wall, component=1),
        dde.icbc.DirichletBC(geom, lambda x: 0, bottom_wall, component=1),
        dde.icbc.OperatorBC(
            geom,
            lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
            bottom_wall,
        ),
        dde.icbc.OperatorBC(
            geom,
            lambda x, y, _: dde.grad.jacobian(y, x, i=2, j=1),
            bottom_wall,
        ),
        dde.icbc.DirichletBC(geom, lambda x: 1, left_wall, component=0),
        dde.icbc.DirichletBC(geom, lambda x: 0, left_wall, component=1),
        dde.icbc.DirichletBC(geom, lambda x: 0, right_wall, component=2),
    ]

    data = dde.data.PDE(
        geom,
        physLoss,
        bcs,
        num_domain=INTERIOR_PTS,
        num_boundary=BOUNDARY_PTS,
    )
    net = dde.nn.FNN(LAYER_SIZE, ACTIVATION, INITIALIZER)
    model = dde.Model(data, net)
    # Compile so the restored model has the same output/network setup as training.
    model.compile("adam", lr=0.001, loss_weights=LOSS_WEIGHTS, verbose=0)
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def error_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.mean(np.abs(values))),
        "max_abs": float(np.max(np.abs(values))) if values.size else float("nan"),
    }


def find_checkpoint(path_text: str) -> Path:
    path = Path(path_text)
    candidates = [path]
    if path.suffix != ".pt":
        candidates.append(Path(str(path) + ".pt"))
    else:
        candidates.append(path.with_suffix(""))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find checkpoint '{path_text}'. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_inference(model, points: np.ndarray, batch_sizes: list[int], repeats: int) -> list[dict[str, Any]]:
    device = next(model.net.parameters()).device
    results = []
    for requested_size in batch_sizes:
        size = min(int(requested_size), len(points))
        batch = points[:size]
        # Warm-up avoids counting first-call setup overhead in the timed samples.
        for _ in range(2):
            model.predict(batch)
        timings_ms = []
        for _ in range(repeats):
            sync_device(device)
            start = time.perf_counter()
            model.predict(batch)
            sync_device(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
        median_ms = float(statistics.median(timings_ms))
        results.append({
            "batch_size": size,
            "repeats": repeats,
            "median_ms": median_ms,
            "p95_ms": float(np.percentile(timings_ms, 95)),
            "points_per_second": float(size / (median_ms / 1000.0)) if median_ms > 0 else float("inf"),
            "all_timings_ms": [float(x) for x in timings_ms],
        })
    return results


def network_outputs_and_derivatives(
    model, points: np.ndarray, second_order: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Evaluate the network and its input derivatives with PyTorch autograd.

    This intentionally bypasses DeepXDE's ``model.predict(..., operator=...)``:
    some DeepXDE versions expect ``data.auxiliary_var_fn`` to be callable even
    when the model has no auxiliary variables.
    """
    device = next(model.net.parameters()).device
    dtype = next(model.net.parameters()).dtype
    x = torch.as_tensor(points, dtype=dtype, device=device).clone().detach()
    x.requires_grad_(True)
    y = model.net(x)

    first = []
    for output_index in range(y.shape[1]):
        grad_y = torch.autograd.grad(
            y[:, output_index].sum(), x, create_graph=True, retain_graph=True
        )[0]
        first.append(grad_y)
    first_tensor = torch.stack(first, dim=1)  # [samples, outputs, input dimensions]

    second_tensor = None
    if second_order:
        second_outputs = []
        for output_index in range(y.shape[1]):
            second_inputs = []
            for input_index in range(x.shape[1]):
                grad_component = first_tensor[:, output_index, input_index]
                grad2 = torch.autograd.grad(
                    grad_component.sum(), x, create_graph=False, retain_graph=True
                )[0]
                second_inputs.append(grad2)
            second_outputs.append(torch.stack(second_inputs, dim=1))
        second_tensor = torch.stack(second_outputs, dim=1)

    return (
        y.detach().cpu().numpy(),
        first_tensor.detach().cpu().numpy(),
        None if second_tensor is None else second_tensor.detach().cpu().numpy(),
    )


def evaluate_pde_residuals(model, points: np.ndarray) -> np.ndarray:
    """Compute the three PDE residuals without DeepXDE's operator prediction API."""
    device = next(model.net.parameters()).device
    dtype = next(model.net.parameters()).dtype
    x = torch.as_tensor(points, dtype=dtype, device=device).clone().detach()
    x.requires_grad_(True)
    y = model.net(x)

    first = []
    for output_index in range(3):
        first.append(torch.autograd.grad(
            y[:, output_index].sum(), x, create_graph=True, retain_graph=True
        )[0])

    # First derivatives: output index, then z/r input derivative.
    duz_z, duz_r = first[0][:, 0], first[0][:, 1]
    dur_z, dur_r = first[1][:, 0], first[1][:, 1]
    dp_z, dp_r = first[2][:, 0], first[2][:, 1]

    # Required second derivatives of axial and radial velocity.
    duz_zz = torch.autograd.grad(duz_z.sum(), x, create_graph=True, retain_graph=True)[0][:, 0]
    duz_rr = torch.autograd.grad(duz_r.sum(), x, create_graph=True, retain_graph=True)[0][:, 1]
    dur_zz = torch.autograd.grad(dur_z.sum(), x, create_graph=True, retain_graph=True)[0][:, 0]
    dur_rr = torch.autograd.grad(dur_r.sum(), x, create_graph=True, retain_graph=True)[0][:, 1]

    r = x[:, 1]
    uz, ur = y[:, 0], y[:, 1]
    nu_star = 2.0 / RE
    loss_z_mom = r * (ur * duz_r + uz * duz_z + dp_z) - nu_star * (
        r * (duz_rr + duz_zz) + duz_r
    )
    loss_r_mom = r**2 * (ur * dur_r + uz * dur_z + dp_r) - nu_star * (
        r**2 * (dur_rr + dur_zz) + r * dur_r - ur
    )
    loss_cont = r * (dur_r + duz_z) + ur
    return torch.stack((loss_z_mom, loss_r_mom, loss_cont), dim=1).detach().cpu().numpy()


def evaluate_boundary_errors(model, rng: np.random.Generator, n: int) -> dict[str, Any]:
    # Exclude corners because multiple boundary conditions meet there.
    z_interior = rng.uniform(1e-4, L_R - 1e-4, n)
    r_interior = rng.uniform(1e-4, 1.0 - 1e-4, n)
    r_axis = np.zeros((n, 2))
    r_axis[:, 0] = z_interior
    r_axis[:, 1] = 0.0

    r_top = np.column_stack((z_interior, np.ones(n)))
    r_inlet = np.column_stack((np.zeros(n), r_interior))
    r_outlet = np.column_stack((np.full(n, L_R), r_interior))

    pred_top = model.predict(r_top)
    pred_axis = model.predict(r_axis)
    pred_inlet = model.predict(r_inlet)
    pred_outlet = model.predict(r_outlet)

    _, axis_derivatives, _ = network_outputs_and_derivatives(model, r_axis)
    derivative_uz_r = axis_derivatives[:, 0, 1]
    derivative_p_r = axis_derivatives[:, 2, 1]

    return {
        "top_wall_u_z_equals_0": error_metrics(pred_top[:, 0]),
        "top_wall_u_r_equals_0": error_metrics(pred_top[:, 1]),
        "axis_u_r_equals_0": error_metrics(pred_axis[:, 1]),
        "axis_du_z_dr_equals_0": error_metrics(derivative_uz_r),
        "axis_dp_dr_equals_0": error_metrics(derivative_p_r),
        "inlet_u_z_equals_1": error_metrics(pred_inlet[:, 0] - 1.0),
        "inlet_u_r_equals_0": error_metrics(pred_inlet[:, 1]),
        "outlet_p_equals_0": error_metrics(pred_outlet[:, 2]),
    }


def evaluate_model(
    label: str,
    checkpoint: Path,
    plot_points: np.ndarray,
    validation_points: np.ndarray,
    output_dir: Path,
    batch_sizes: list[int],
    repeats: int,
    boundary_samples: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    print(f"\n{'=' * 72}\nLoading {label}: {checkpoint}\n{'=' * 72}")
    model = build_model()

    # This is an evaluation-only script, so restore network weights directly.
    # DeepXDE's model.restore() also restores the optimizer state, which is not
    # needed for inference and can fail if an older checkpoint has an
    # incompatible Adam state (for example, a missing "step" entry).
    device = next(model.net.parameters()).device
    try:
        saved_state = torch.load(
            str(checkpoint), map_location=device, weights_only=False
        )
    except TypeError:
        # Compatibility with PyTorch versions without the weights_only argument.
        saved_state = torch.load(str(checkpoint), map_location=device)

    if isinstance(saved_state, dict) and "model_state_dict" in saved_state:
        model.net.load_state_dict(saved_state["model_state_dict"])
    elif isinstance(saved_state, dict) and saved_state and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in saved_state.items()
    ):
        # Also accept a checkpoint that contains only a plain network state dict.
        model.net.load_state_dict(saved_state)
    else:
        keys = list(saved_state.keys()) if isinstance(saved_state, dict) else type(saved_state).__name__
        raise ValueError(
            f"Unsupported checkpoint format for {checkpoint}. "
            f"Expected 'model_state_dict' or a plain state dict; found: {keys}"
        )

    model.net.eval()
    device = next(model.net.parameters()).device

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    predictions = model.predict(plot_points)
    validation_predictions = model.predict(validation_points)

    residuals = evaluate_pde_residuals(model, validation_points)
    pde_metrics = {
        name: error_metrics(residuals[:, i])
        for i, name in enumerate(["axial_momentum", "radial_momentum", "continuity"])
    }

    # Use a fresh deterministic RNG per model for equivalent boundary locations.
    boundary_metrics = evaluate_boundary_errors(
        model, np.random.default_rng(20261015), boundary_samples
    )
    timing_metrics = benchmark_inference(model, plot_points, batch_sizes, repeats)

    gpu_memory_mb = None
    if device.type == "cuda":
        sync_device(device)
        gpu_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))

    try:
        parameter_count = int(sum(p.numel() for p in model.net.parameters()))
    except Exception:
        parameter_count = None

    result = {
        "label": label,
        "checkpoint": str(checkpoint),
        "model_parameters": parameter_count,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "deepxde": getattr(dde, "__version__", "unknown"),
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_allocated_mb": gpu_memory_mb,
        },
        "inference_timing": timing_metrics,
        "pde_residuals": pde_metrics,
        "boundary_errors": boundary_metrics,
        "validation_prediction_finite": bool(np.isfinite(validation_predictions).all()),
        "plot_prediction_finite": bool(np.isfinite(predictions).all()),
    }

    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    np.savez_compressed(
        output_dir / f"{safe_label}_predictions.npz",
        plot_points=plot_points,
        plot_predictions=predictions,
        validation_points=validation_points,
        validation_predictions=validation_predictions,
        validation_residuals=residuals,
    )
    (output_dir / f"{safe_label}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(f"Device: {device} | Parameters: {parameter_count}")
    print("PDE residual RMSE:")
    for name, values in pde_metrics.items():
        print(f"  {name:18s} {values['rmse']:.6e}")
    print("Boundary-condition RMSE:")
    for name, values in boundary_metrics.items():
        print(f"  {name:28s} {values['rmse']:.6e}")
    print("Inference timing (median; lower is faster):")
    for item in timing_metrics:
        print(
            f"  n={item['batch_size']:>6,d}: {item['median_ms']:>9.3f} ms, "
            f"p95={item['p95_ms']:.3f} ms, "
            f"{item['points_per_second']:,.0f} points/s"
        )
    print(f"Saved metrics: {output_dir / f'{safe_label}_metrics.json'}")
    print(f"Saved predictions: {output_dir / f'{safe_label}_predictions.npz'}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, predictions, validation_predictions


def prediction_difference(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    output = {}
    for i, name in enumerate(["u_z", "u_r", "p"]):
        difference = b[:, i] - a[:, i]
        denom = float(np.linalg.norm(a[:, i]))
        output[name] = {
            "rmse_difference": float(np.sqrt(np.mean(difference**2))),
            "mae_difference": float(np.mean(np.abs(difference))),
            "max_abs_difference": float(np.max(np.abs(difference))),
            "relative_l2_difference": float(np.linalg.norm(difference) / denom) if denom > 1e-12 else None,
        }
    return output



def plot_model_comparison(
    original_metrics: dict[str, Any],
    experimental_metrics: dict[str, Any],
    plots_dir: Path,
) -> Path:
    """Save a compact visual comparison of speed and physics consistency."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Pretrained Fuel-Cell PINN Comparison", fontsize=14)

    # 1) Inference latency: lower is better.
    original_timing = original_metrics["inference_timing"]
    experimental_timing = experimental_metrics["inference_timing"]
    original_sizes = [item["batch_size"] for item in original_timing]
    experimental_sizes = [item["batch_size"] for item in experimental_timing]
    axes[0].plot(original_sizes, [item["median_ms"] for item in original_timing],
                 marker="o", label="Original")
    axes[0].plot(experimental_sizes, [item["median_ms"] for item in experimental_timing],
                 marker="o", label="Experimental")
    axes[0].set_title("Inference Latency")
    axes[0].set_xlabel("Batch size (points)")
    axes[0].set_ylabel("Median time (ms) — lower is better")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2) PDE residual RMSE: separate equations, logarithmic scale.
    pde_names = ["axial_momentum", "radial_momentum", "continuity"]
    pde_labels = ["Axial momentum", "Radial momentum", "Continuity"]
    x = np.arange(len(pde_names))
    width = 0.36
    orig_pde = [original_metrics["pde_residuals"][name]["rmse"] for name in pde_names]
    experimental_pde = [experimental_metrics["pde_residuals"][name]["rmse"] for name in pde_names]
    axes[1].bar(x - width / 2, np.maximum(orig_pde, 1e-16), width, label="Original")
    axes[1].bar(x + width / 2, np.maximum(experimental_pde, 1e-16), width, label="Experimental")
    axes[1].set_title("PDE Residual RMSE")
    axes[1].set_ylabel("RMSE — lower is better (log scale)")
    axes[1].set_xticks(x, pde_labels, rotation=20, ha="right")
    axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    # 3) Boundary-condition RMSE, grouped by condition.
    boundary_names = list(original_metrics["boundary_errors"].keys())
    boundary_labels = [name.replace("_equals_", " = ").replace("_", " ")
                       for name in boundary_names]
    orig_boundary = [original_metrics["boundary_errors"][name]["rmse"]
                     for name in boundary_names]
    experimental_boundary = [experimental_metrics["boundary_errors"][name]["rmse"]
                     for name in boundary_names]
    xb = np.arange(len(boundary_names))
    axes[2].bar(xb - width / 2, np.maximum(orig_boundary, 1e-16), width, label="Original")
    axes[2].bar(xb + width / 2, np.maximum(experimental_boundary, 1e-16), width, label="Experimental")
    axes[2].set_title("Boundary-Condition Error")
    axes[2].set_ylabel("RMSE — lower is better (log scale)")
    axes[2].set_xticks(xb, boundary_labels, rotation=65, ha="right", fontsize=8)
    axes[2].set_yscale("log")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend()

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path = plots_dir / "model_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path

def main() -> None:
    dde.config.set_random_seed(SEED)

    # Create output directories (and any missing parent directories).
    output_dir = OUTPUT_DIRECTORY
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    original_checkpoint = find_checkpoint(str(ORIGINAL_MODEL_PATH))
    experimental_checkpoint = find_checkpoint(str(EXPERIMENTAL_MODEL_PATH))

    # Identical deterministic locations for both models.
    plot_axis = np.linspace(0.0, L_R, 200)
    radial_axis = np.linspace(0.0, 1.0, 200)
    Z, RR = np.meshgrid(plot_axis, radial_axis)
    plot_points = np.column_stack((Z.ravel(), RR.ravel()))
    rng = np.random.default_rng(20261014)
    validation_points = np.column_stack((
        rng.uniform(0.0, L_R, 10000),
        rng.uniform(1e-4, 1.0 - 1e-4, 10000),
    ))

    print("PINN pretrained-model comparison")
    print(f"Domain: z in [0, {L_R:.6f}], r in [0, 1]")
    print(f"Validation points: {len(validation_points):,}; plot-grid points: {len(plot_points):,}")
    print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    batch_sizes = [1, 100, 1000, 10000, len(plot_points)]
    original_metrics, original_plot, original_validation = evaluate_model(
        "original", original_checkpoint, plot_points, validation_points, output_dir,
        batch_sizes, INFERENCE_REPEATS, BOUNDARY_SAMPLES,
    )
    experimental_metrics, experimental_plot, experimental_validation = evaluate_model(
        "experimental", experimental_checkpoint, plot_points, validation_points, output_dir,
        batch_sizes, INFERENCE_REPEATS, BOUNDARY_SAMPLES,
    )

    comparison = {
        "original_checkpoint": str(original_checkpoint),
        "experimental_checkpoint": str(experimental_checkpoint),
        "same_validation_points": True,
        "validation_prediction_differences_experimental_minus_original": prediction_difference(
            original_validation, experimental_validation
        ),
        "plot_prediction_differences_experimental_minus_original": prediction_difference(
            original_plot, experimental_plot
        ),
        "notes": [
            "Lower per-equation PDE residuals indicate better consistency with the encoded equations, not necessarily accuracy against a real solution.",
            "Boundary metrics are errors against the boundary conditions encoded in this script.",
            "Inference timing includes the DeepXDE model.predict call overhead and depends on hardware/software load.",
            "For physical accuracy, compare against trusted simulation or experimental reference data when available.",
        ],
    }
    comparison_path = output_dir / "model_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    plot_path = plot_model_comparison(original_metrics, experimental_metrics, plots_dir)
    print(f"Comparison plot saved to: {plot_path.resolve()}")

    print(f"\n{'=' * 72}\nPrediction differences (experimental minus original)\n{'=' * 72}")
    for name, values in comparison["validation_prediction_differences_experimental_minus_original"].items():
        print(
            f"{name}: RMSE={values['rmse_difference']:.6e}, "
            f"MAE={values['mae_difference']:.6e}, "
            f"max={values['max_abs_difference']:.6e}, "
            f"relative L2={values['relative_l2_difference']}"
        )
    print(f"\nComplete comparison saved to: {comparison_path.resolve()}")
    print("No training was performed; checkpoints were only loaded and evaluated.")


if __name__ == "__main__":
    main()
