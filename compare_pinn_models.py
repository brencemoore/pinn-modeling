"""Compare two pretrained DeepXDE/PyTorch fuel-cell PINNs of any model version.

Edit the configuration variables below, then run this file in the
pinn-modeling conda environment.

This script never trains a model. Both checkpoints must already exist in the
models/<version>/ folders; if either is missing (or both point to the same file)
the script lists the available checkpoints and exits.

Each run is saved to model_comparisons/<A>_vs_<B>_<date_time>/:
  plots/      comparison figures
  metrics/    per-model metrics and predictions, and model_comparison.json
  README.md   which two checkpoints were compared, and when

The physics of each model is read from that model's own training script (see
pinn_utils.py), so the comparison stays in sync with the training code. To
support a new model version, add its training script to MODEL_SCRIPTS in
pinn_utils.py.

Results focus on:
  - Accuracy: PDE residuals, boundary-condition errors, and each model's own
    weighted training objective, all evaluated on fixed points.
  - Computation cost: parameters, FLOPs per point, inference time, PDE-residual
    evaluation time (a proxy for the cost of one training step), training steps,
    and training time (when the checkpoint has a training record).
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pinn_utils as pu


# ---------------------------------------------------------------------------
# User configuration: edit these before running the script.
# Paths are relative to the project root (the working directory in VS Code).
# ---------------------------------------------------------------------------
MODEL_A_PATH = Path("models/v3/pinn_model_V3_2026-10-28_0526-7000.pt")
MODEL_B_PATH = Path("models/v3_exp/pinn_model_V3_exp_2026-10-28_1322-7000.pt")
MODEL_A_LABEL = "V3"
MODEL_B_LABEL = "V3_exp"

# Set to a script path to skip the automatic lookup (needed for older
# checkpoints saved without a version, e.g. pinn_model-10000.pt)
MODEL_A_SCRIPT = None
MODEL_B_SCRIPT = None

# Each run gets its own folder: model_comparisons/<A>_vs_<B>_<date_time>/
OUTPUT_DIRECTORY = Path("model_comparisons")
VALIDATION_POINTS = 10000
BOUNDARY_SAMPLES = 1000      # points sampled along each edge of the domain
GRID_SIZE = 200
INFERENCE_REPEATS = 15
RESIDUAL_REPEATS = 3


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def require_two_checkpoints() -> tuple[Path, Path]:
    """Exit unless both configured checkpoints exist and are different files."""
    checkpoint_a = pu.resolve_checkpoint(MODEL_A_PATH)
    checkpoint_b = pu.resolve_checkpoint(MODEL_B_PATH)

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
        available = pu.available_checkpoints()
        print("\nAvailable checkpoints in models/<version>/ (oldest first):")
        for path in available:
            print(f"  {path}")
        if not available:
            print("  (none)")
        print("\nSet MODEL_A_PATH and MODEL_B_PATH to two different checkpoints. "
              "This script does not train models.")
        sys.exit(1)

    return checkpoint_a, checkpoint_b


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def timings_ms(function, device: torch.device, repeats: int) -> list[float]:
    # Warm-up avoids counting first-call setup overhead in the timed samples
    function()
    timings = []
    for _ in range(repeats):
        pu.sync_device(device)
        start = time.perf_counter()
        function()
        pu.sync_device(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def benchmark_inference(net: torch.nn.Module, points: np.ndarray, repeats: int) -> list[dict[str, Any]]:
    device = next(net.parameters()).device
    results = []
    for size in sorted({1, 100, 1000, 10000, len(points)}):
        batch = points[:min(size, len(points))]
        timings = timings_ms(lambda: pu.predict(net, batch), device, repeats)
        median = float(statistics.median(timings))
        results.append({
            "batch_size": len(batch),
            "median_ms": median,
            "p95_ms": float(np.percentile(timings, 95)),
            "points_per_second": float(len(batch) / (median / 1000.0)) if median > 0 else None,
        })
    return results


def evaluate_model(label: str, checkpoint: Path, script: str | None,
                   device: torch.device) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    print(f"\n{'=' * 72}\nEvaluating {label}: {checkpoint}\n{'=' * 72}")
    model = pu.load_model(checkpoint, device, script)
    info, geom = model.info, model.definitions["geom"]
    print(f"Physics from: {info['script']}")

    Z, _, grid_points = pu.evaluation_grid(geom, GRID_SIZE)
    validation_points = pu.validation_points(geom, VALIDATION_POINTS)

    grid_predictions = pu.predict(model.net, grid_points)
    validation_predictions = pu.predict(model.net, validation_points)
    validation_residuals = pu.pde_residuals(model, validation_points)
    grid_residuals = pu.pde_residuals(model, grid_points)

    info["equations"] = pu.equation_names(model, validation_residuals)
    info["boundary_conditions"] = model.definitions["bc_names"]
    pde_metrics = {name: pu.error_metrics(validation_residuals[:, i]) for i, name in enumerate(info["equations"])}
    bc_metrics = pu.boundary_errors(model, pu.boundary_points(geom, BOUNDARY_SAMPLES))

    objective = pu.training_objective(model, pde_metrics, bc_metrics)
    if objective is None and model.definitions["weights"] is not None:
        print("Note: loss weights do not match the loss terms; training objective not computed.")

    residual_timings = timings_ms(
        lambda: pu.pde_residuals(model, validation_points), device, RESIDUAL_REPEATS)
    record = model.training_record

    result = {
        "label": label,
        "model": info,
        "accuracy": {
            "training_objective": objective,
            "loss_weights": model.definitions["weights"],
            "mean_pde_rmse": float(np.mean([m["rmse"] for m in pde_metrics.values()])),
            "pde_residuals": pde_metrics,
            "boundary_errors": bc_metrics,
        },
        "cost": {
            "parameters": info["parameters"],
            "forward_flops_per_point": pu.forward_flops(info["layer_size"]),
            "training_steps": info["step"],
            "lbfgs_iterations": info["lbfgs_iterations"],
            "training_seconds": record["training_seconds"] if record else None,
            "residual_ms_per_1000_points": float(statistics.median(residual_timings))
                                           * 1000.0 / len(validation_points),
            "inference_timing": benchmark_inference(model.net, grid_points, INFERENCE_REPEATS),
        },
        "predictions_finite": bool(np.isfinite(grid_predictions).all()
                                   and np.isfinite(validation_predictions).all()),
    }

    print(f"Version: {info['version']} | Trained: {info['trained']} | Step: {info['step']} "
          f"| L-BFGS iterations: {info['lbfgs_iterations']}")
    print(f"Layers: {info['layer_size']} | Parameters: {info['parameters']:,} | dtype: {info['dtype']}")
    print(f"Outputs: {info['outputs']} | Equations: {info['equations']}")
    print("Training record: " + (f"{record['training_seconds'] / 60:.1f} min" if record else "none"))
    if not result["predictions_finite"]:
        print("WARNING: predictions contain NaN or infinity.")

    arrays = {
        "grid_points": grid_points,
        "grid_shape": np.array(Z.shape),
        "grid_predictions": grid_predictions,
        "grid_residuals": grid_residuals,
        "validation_points": validation_points,
        "validation_predictions": validation_predictions,
        "validation_residuals": validation_residuals,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, arrays


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def same_physics(result_a: dict[str, Any], result_b: dict[str, Any]) -> bool:
    a, b = result_a["model"], result_b["model"]
    return all(a[key] == b[key] for key in ("outputs", "equations", "boundary_conditions", "domain"))


def shared_outputs(result_a: dict[str, Any], result_b: dict[str, Any]) -> list[str]:
    """Outputs both models predict on the same domain (needed to compare fields)."""
    if result_a["model"]["domain"] != result_b["model"]["domain"]:
        return []
    return [name for name in result_a["model"]["outputs"] if name in result_b["model"]["outputs"]]


def prediction_difference(result_a, result_b, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    output = {}
    for name in shared_outputs(result_a, result_b):
        i = result_a["model"]["outputs"].index(name)
        j = result_b["model"]["outputs"].index(name)
        difference = b[:, j] - a[:, i]
        denominator = float(np.linalg.norm(a[:, i]))
        output[name] = {
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "max_abs": float(np.max(np.abs(difference))),
            "relative_l2": float(np.linalg.norm(difference) / denominator) if denominator > 1e-12 else None,
        }
    return output


def summary_rows(result_a: dict[str, Any], result_b: dict[str, Any]) -> list[dict[str, Any]]:
    """Scalar metrics side by side, where lower is better for every row."""
    acc_a, acc_b = result_a["accuracy"], result_b["accuracy"]
    cost_a, cost_b = result_a["cost"], result_b["cost"]

    rows = [
        ("accuracy", "Training objective (weighted loss)", acc_a["training_objective"], acc_b["training_objective"]),
        ("accuracy", "Mean PDE residual RMSE", acc_a["mean_pde_rmse"], acc_b["mean_pde_rmse"]),
    ]
    # Per-equation and per-BC rows only where both models have the same term
    for name in acc_a["pde_residuals"]:
        if name in acc_b["pde_residuals"]:
            rows.append(("accuracy", f"PDE {name} RMSE",
                         acc_a["pde_residuals"][name]["rmse"], acc_b["pde_residuals"][name]["rmse"]))
    for name, metrics in acc_a["boundary_errors"].items():
        other = acc_b["boundary_errors"].get(name)
        if metrics and other:
            rows.append(("accuracy", f"BC {name} RMSE", metrics["rmse"], other["rmse"]))

    largest_a, largest_b = cost_a["inference_timing"][-1], cost_b["inference_timing"][-1]
    rows += [
        ("cost", "Parameters", cost_a["parameters"], cost_b["parameters"]),
        ("cost", "Forward FLOPs per point", cost_a["forward_flops_per_point"], cost_b["forward_flops_per_point"]),
        ("cost", "PDE residual ms per 1k points", cost_a["residual_ms_per_1000_points"],
         cost_b["residual_ms_per_1000_points"]),
        ("cost", f"Inference ms (n={largest_a['batch_size']:,} vs {largest_b['batch_size']:,})",
         largest_a["median_ms"], largest_b["median_ms"]),
        ("cost", "Training steps", cost_a["training_steps"], cost_b["training_steps"]),
        ("cost", "Training time (s)", cost_a["training_seconds"], cost_b["training_seconds"]),
    ]

    output = []
    for kind, metric, a, b in rows:
        if a is None or b is None:
            continue
        change = (b - a) / a * 100.0 if a else None
        better = "tie" if np.isclose(a, b, rtol=1e-3, atol=0) else ("A" if a < b else "B")
        output.append({"kind": kind, "metric": metric, "a": a, "b": b,
                       "change_percent": change, "better": better})
    return output


def print_summary(rows: list[dict[str, Any]], differences: dict[str, Any], physics_match: bool) -> None:
    print(f"\n{'=' * 78}\nSummary: A = {MODEL_A_LABEL}, B = {MODEL_B_LABEL} (lower is better)\n{'=' * 78}")
    if not physics_match:
        print("Note: the models solve different physics. Accuracy is each model measured "
              "against its own equations;\nonly computation cost is directly comparable.\n")
    for kind, title in (("accuracy", "Accuracy"), ("cost", "Computation cost")):
        kind_rows = [row for row in rows if row["kind"] == kind]
        print(f"{title:<44} {'A':>10} {'B':>10} {'B vs A':>9}  Better")
        for row in kind_rows:
            change = f"{row['change_percent']:+.1f}%" if row["change_percent"] is not None else "--"
            print(f"  {row['metric']:<42} {row['a']:>10.3e} {row['b']:>10.3e} {change:>9}  {row['better']}")
        wins_a = sum(row["better"] == "A" for row in kind_rows)
        wins_b = sum(row["better"] == "B" for row in kind_rows)
        print(f"  Lower on {wins_a} {kind} metrics: A | Lower on {wins_b}: B\n")

    if differences:
        print("Prediction differences on validation points (B minus A):")
        for name, values in differences.items():
            relative = f"{values['relative_l2']:.3%}" if values["relative_l2"] is not None else "--"
            print(f"  {name:<4} RMSE={values['rmse']:.3e}  max={values['max_abs']:.3e}  relative L2={relative}")
    else:
        print("No shared outputs on the same domain; predicted fields are not compared directly.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def grouped_bars(ax, names_a: dict, names_b: dict, title: str) -> None:
    names = list(names_a) + [n for n in names_b if n not in names_a]
    x = np.arange(len(names))
    width = 0.36
    for offset, values, label in ((-width / 2, names_a, MODEL_A_LABEL), (width / 2, names_b, MODEL_B_LABEL)):
        heights = [max(values[n]["rmse"], 1e-16) if values.get(n) else np.nan for n in names]
        ax.bar(x + offset, heights, width, label=label)
    ax.set_title(title)
    ax.set_ylabel("RMSE (log scale), lower is better")
    ax.set_xticks(x, [n.replace("_", " ") for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()


def plot_metrics(result_a: dict[str, Any], result_b: dict[str, Any], plots_dir: Path) -> Path:
    """Accuracy (residuals, boundary errors) and computation cost at a glance."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"PINN Comparison: {MODEL_A_LABEL} vs {MODEL_B_LABEL}", fontsize=14)

    grouped_bars(axes[0, 0], result_a["accuracy"]["pde_residuals"],
                 result_b["accuracy"]["pde_residuals"], "Accuracy: PDE Residual")
    grouped_bars(axes[0, 1], result_a["accuracy"]["boundary_errors"],
                 result_b["accuracy"]["boundary_errors"], "Accuracy: Boundary-Condition Error")

    for result, label in ((result_a, MODEL_A_LABEL), (result_b, MODEL_B_LABEL)):
        timing = result["cost"]["inference_timing"]
        axes[1, 0].plot([t["batch_size"] for t in timing], [t["median_ms"] for t in timing],
                        marker="o", label=label)
    axes[1, 0].set_title("Cost: Inference Latency")
    axes[1, 0].set_xlabel("Batch size (points)")
    axes[1, 0].set_ylabel("Median time (ms), lower is better")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Cost of B relative to A, so metrics with different units share one axis
    cost_names = {
        "parameters": "Parameters",
        "forward_flops_per_point": "FLOPs/point",
        "residual_ms_per_1000_points": "PDE residual time",
        "training_steps": "Training steps",
        "training_seconds": "Training time",
    }
    labels, ratios = [], []
    for key, name in cost_names.items():
        a, b = result_a["cost"][key], result_b["cost"][key]
        if a and b is not None:
            labels.append(name)
            ratios.append(b / a)
    largest_a = result_a["cost"]["inference_timing"][-1]["median_ms"]
    largest_b = result_b["cost"]["inference_timing"][-1]["median_ms"]
    labels.append("Inference time")
    ratios.append(largest_b / largest_a)
    colors = ["tab:green" if r < 1 else "tab:red" for r in ratios]
    axes[1, 1].barh(labels, ratios, color=colors)
    axes[1, 1].axvline(1.0, color="black", linewidth=1)
    axes[1, 1].set_title(f"Cost: {MODEL_B_LABEL} relative to {MODEL_A_LABEL}")
    axes[1, 1].set_xlabel(f"{MODEL_B_LABEL} / {MODEL_A_LABEL} (below 1 = cheaper)")
    axes[1, 1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = plots_dir / "model_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def grid_axes(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(arrays["grid_shape"])
    return arrays["grid_points"][:, 0].reshape(shape), arrays["grid_points"][:, 1].reshape(shape)


def plot_predicted_fields(result_a, result_b, arrays_a, arrays_b, plots_dir: Path) -> Path:
    """Each model's predicted outputs side by side (works for any model types)."""
    rows = max(len(result_a["model"]["outputs"]), len(result_b["model"]["outputs"]))
    fig, axes = plt.subplots(rows, 2, figsize=(16, 2.6 * rows + 0.6), squeeze=False)
    fig.suptitle("Predicted fields")
    for column, (result, arrays) in enumerate(((result_a, arrays_a), (result_b, arrays_b))):
        Z, RR = grid_axes(arrays)
        outputs = result["model"]["outputs"]
        for row in range(rows):
            ax = axes[row, column]
            if row >= len(outputs):
                ax.axis("off")
                continue
            values = arrays["grid_predictions"][:, row].reshape(Z.shape)
            contour = ax.contourf(Z, RR, values, levels=100, cmap="jet")
            fig.colorbar(contour, ax=ax, label=outputs[row])
            ax.set_title(f"{result['label']}: {outputs[row]}")
            ax.set_ylabel("r")
        axes[-1, column].set_xlabel("z")
    fig.tight_layout()
    path = plots_dir / "predicted_fields.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_field_differences(result_a, result_b, arrays_a, arrays_b, plots_dir: Path) -> Path | None:
    """Contours of where shared predicted fields disagree (B minus A)."""
    names = shared_outputs(result_a, result_b)
    if not names:
        return None
    Z, RR = grid_axes(arrays_a)
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.4 * len(names) + 0.6), sharex=True, squeeze=False)
    fig.suptitle(f"Prediction difference: {MODEL_B_LABEL} minus {MODEL_A_LABEL}")
    for ax, name in zip(axes[:, 0], names):
        i = result_a["model"]["outputs"].index(name)
        j = result_b["model"]["outputs"].index(name)
        difference = (arrays_b["grid_predictions"][:, j] - arrays_a["grid_predictions"][:, i]).reshape(Z.shape)
        # 99.5th percentile keeps a single corner spike from washing out the colors
        limit = max(float(np.percentile(np.abs(difference), 99.5)), 1e-12)
        contour = ax.contourf(Z, RR, np.clip(difference, -limit, limit),
                              levels=np.linspace(-limit, limit, 101), cmap="RdBu_r", extend="both")
        fig.colorbar(contour, ax=ax, label=f"Δ{name}")
        ax.set_ylabel("r")
    axes[-1, 0].set_xlabel("z")
    fig.tight_layout()
    path = plots_dir / "field_differences.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residual_maps(result_a, result_b, arrays_a, arrays_b, plots_dir: Path, physics_match: bool) -> Path:
    """Local PDE residual of each model (shared color scale when the physics match)."""
    local = []
    for arrays in (arrays_a, arrays_b):
        Z, RR = grid_axes(arrays)
        squared = np.sum(arrays["grid_residuals"].astype(np.float64)**2, axis=1)
        local.append((Z, RR, np.maximum(squared, 1e-20).reshape(Z.shape)))
    shared_norm = None
    if physics_match:
        shared_norm = LogNorm(vmin=min(v.min() for _, _, v in local), vmax=max(v.max() for _, _, v in local))

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), constrained_layout=True)
    for ax, (Z, RR, values), result in zip(axes, local, (result_a, result_b)):
        norm = shared_norm or LogNorm(vmin=values.min(), vmax=values.max())
        mesh = ax.pcolormesh(Z, RR, values, shading="auto", cmap="inferno", norm=norm)
        ax.set_title(f"Local PDE residual: {result['label']}")
        ax.set_ylabel("r")
        if not physics_match:
            fig.colorbar(mesh, ax=ax, label="Sum of squared residuals")
    axes[-1].set_xlabel("z")
    if physics_match:
        fig.colorbar(mesh, ax=axes, label="Sum of squared PDE residuals")
    path = plots_dir / "residual_maps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_run_summary(path: Path, run_time: datetime, result_a: dict[str, Any],
                      result_b: dict[str, Any], physics_match: bool) -> None:
    """Small note recording which checkpoints were compared and when."""
    lines = ["# Model Comparison", "", f"- Run: {run_time:%Y-%m-%d %H:%M:%S}"]
    for name, result in (("A", result_a), ("B", result_b)):
        model = result["model"]
        lines.append(f"- Model {name} ({result['label']}): `{Path(model['checkpoint']).as_posix()}` "
                     f"(physics: `{Path(model['script']).name}`)")
    lines.append(f"- Same physics: {'yes' if physics_match else 'no'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Check checkpoints and scripts before doing any other work
    checkpoint_a, checkpoint_b = require_two_checkpoints()
    try:
        pu.resolve_script(checkpoint_a, MODEL_A_SCRIPT)
        pu.resolve_script(checkpoint_b, MODEL_B_SCRIPT)
    except pu.PinnModelError as error:
        sys.exit(str(error))

    run_time = datetime.now()
    output_dir = OUTPUT_DIRECTORY / (
        f"{pu.safe_name(MODEL_A_LABEL)}_vs_{pu.safe_name(MODEL_B_LABEL)}_{run_time:%Y-%m-%d_%H%M%S}"
    )
    plots_dir = output_dir / "plots"
    metrics_dir = output_dir / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = pu.default_device()
    print("PINN pretrained-model comparison (no training is performed)")
    print(f"Validation points: {VALIDATION_POINTS:,} | Grid: {GRID_SIZE}x{GRID_SIZE} "
          f"| Boundary points per face: {BOUNDARY_SAMPLES:,}")
    print(f"PyTorch {torch.__version__} | Device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    result_a, arrays_a = evaluate_model(MODEL_A_LABEL, checkpoint_a, MODEL_A_SCRIPT, device)
    result_b, arrays_b = evaluate_model(MODEL_B_LABEL, checkpoint_b, MODEL_B_SCRIPT, device)

    physics_match = same_physics(result_a, result_b)
    rows = summary_rows(result_a, result_b)
    differences = prediction_difference(result_a, result_b, arrays_a["validation_predictions"],
                                        arrays_b["validation_predictions"])
    print_summary(rows, differences, physics_match)

    # Save metrics, predictions, and plots
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "deepxde": getattr(dde, "__version__", "unknown"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    for result, arrays in ((result_a, arrays_a), (result_b, arrays_b)):
        safe_label = pu.safe_name(result["label"])
        (metrics_dir / f"{safe_label}_metrics.json").write_text(
            json.dumps({**result, "runtime": runtime}, indent=2), encoding="utf-8")
        np.savez_compressed(metrics_dir / f"{safe_label}_predictions.npz", **arrays)

    comparison = {
        "model_a": result_a["model"],
        "model_b": result_b["model"],
        "same_physics": physics_match,
        "summary_lower_is_better": rows,
        "validation_prediction_differences_b_minus_a": differences,
        "grid_prediction_differences_b_minus_a": prediction_difference(
            result_a, result_b, arrays_a["grid_predictions"], arrays_b["grid_predictions"]),
        "runtime": runtime,
        "notes": [
            "Accuracy metrics measure consistency with each model's own encoded equations and boundary "
            "conditions, not agreement with a trusted reference solution.",
            "When the models solve different physics, only computation cost is directly comparable.",
            "Training time is only available for checkpoints with a training record (<checkpoint>_training.json).",
            "Timing depends on hardware and system load.",
        ],
    }
    (metrics_dir / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    plot_paths = [
        plot_metrics(result_a, result_b, plots_dir),
        plot_predicted_fields(result_a, result_b, arrays_a, arrays_b, plots_dir),
        plot_field_differences(result_a, result_b, arrays_a, arrays_b, plots_dir),
        plot_residual_maps(result_a, result_b, arrays_a, arrays_b, plots_dir, physics_match),
    ]
    write_run_summary(output_dir / "README.md", run_time, result_a, result_b, physics_match)

    print(f"\nComparison saved to: {output_dir.resolve()}")
    for path in plot_paths:
        if path is not None:
            print(f"Plot saved to: {path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except pu.PinnModelError as error:
        sys.exit(str(error))
