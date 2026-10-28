"""Plot one trained fuel-cell PINN checkpoint (any model version).

Edit the configuration variables below, then run this file in the
pinn-modeling conda environment. The training scripts also call
plot_after_training() automatically once a model is saved.

This script never trains a model. Plots are saved to
plots/<checkpoint name>/ together with metrics.json and a short README.md:

  fields.png            every predicted output over the domain
  profiles.png          radial profiles along z, and axial profiles at the axis and wall
  residual_maps.png     where each PDE residual is large
  boundary_errors.png   error of every boundary condition
  loss_history.png      loss terms during training      (needs a training record)
  training_time.png     wall time of each training stage (needs a training record)
  rar_points.png        points added by RAR             (needs a training record with RAR)

The training record (<checkpoint>_training.json) is written by TrainingLog in
the training scripts; older checkpoints without one get the first four plots.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pinn_utils as pu


# ---------------------------------------------------------------------------
# User configuration: edit these before running the script.
# Paths are relative to the project root (the working directory in VS Code).
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/v3/pinn_model_V3_2026-10-28_0526-7000.pt")
# Set to a script path to skip the automatic lookup (needed for older
# checkpoints saved without a version, e.g. pinn_model-10000.pt)
MODEL_SCRIPT = None

OUTPUT_DIRECTORY = Path("plots")
GRID_SIZE = 200
VALIDATION_POINTS = 10000
BOUNDARY_SAMPLES = 1000                       # points sampled along each edge of the domain
PROFILE_POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)  # fractions of the domain length for radial profiles


@dataclass
class Evaluation:
    """Everything computed from the model that the plots and metrics need."""
    Z: np.ndarray
    R: np.ndarray
    fields: np.ndarray             # (grid points, outputs)
    grid_residuals: np.ndarray     # (grid points, equations)
    outputs: list[str]
    equations: list[str]
    pde_metrics: dict[str, Any]
    bc_metrics: dict[str, Any]
    objective: float | None
    cross_section_averages: dict[str, Any]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def cross_section_average(model: pu.LoadedModel, z: float, samples: int = 400) -> np.ndarray:
    """Area-weighted average of each output over the pipe cross-section at z.

    For an axisymmetric pipe: average = integral(f r dr) / integral(r dr).
    """
    low, high = model.definitions["geom"].bbox
    r = np.linspace(low[1], high[1], samples)
    values = pu.predict(model.net, np.column_stack((np.full(samples, z), r)))
    weights = r[:, None]
    return np.trapezoid(values * weights, r, axis=0) / np.trapezoid(r, r)


def evaluate(model: pu.LoadedModel) -> Evaluation:
    geom = model.definitions["geom"]
    Z, R, grid = pu.evaluation_grid(geom, GRID_SIZE)
    validation = pu.validation_points(geom, VALIDATION_POINTS)

    validation_residuals = pu.pde_residuals(model, validation)
    equations = pu.equation_names(model, validation_residuals)
    pde_metrics = {name: pu.error_metrics(validation_residuals[:, i]) for i, name in enumerate(equations)}
    bc_metrics = pu.boundary_errors(model, pu.boundary_points(geom, BOUNDARY_SAMPLES))

    outputs = model.info["outputs"]
    low, high = geom.bbox
    inlet, outlet = cross_section_average(model, low[0]), cross_section_average(model, high[0])
    averages = {name: {"inlet": float(inlet[i]), "outlet": float(outlet[i])} for i, name in enumerate(outputs)}

    return Evaluation(
        Z=Z, R=R,
        fields=pu.predict(model.net, grid),
        grid_residuals=pu.pde_residuals(model, grid),
        outputs=outputs,
        equations=equations,
        pde_metrics=pde_metrics,
        bc_metrics=bc_metrics,
        objective=pu.training_objective(model, pde_metrics, bc_metrics),
        cross_section_averages=averages,
    )


# ---------------------------------------------------------------------------
# Model plots
# ---------------------------------------------------------------------------
def save_figure(fig, path: Path, dpi: int = 160) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fields(result: Evaluation, title: str, output_dir: Path) -> Path:
    rows = len(result.outputs)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.4 * rows + 0.6), squeeze=False, sharex=True)
    fig.suptitle(f"{title}: predicted fields (non-dimensional)")
    for i, (ax, name) in enumerate(zip(axes[:, 0], result.outputs)):
        values = result.fields[:, i].reshape(result.Z.shape)
        contour = ax.contourf(result.Z, result.R, values, levels=100, cmap="jet")
        fig.colorbar(contour, ax=ax, label=name)
        ax.set_ylabel("r")
    axes[-1, 0].set_xlabel("z")
    fig.tight_layout()
    return save_figure(fig, output_dir / "fields.png")


def plot_profiles(result: Evaluation, title: str, output_dir: Path) -> Path:
    """Left: radial profiles at several z. Right: axial profiles at the axis and the wall."""
    rows = len(result.outputs)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.8 * rows + 0.6), squeeze=False)
    fig.suptitle(f"{title}: profiles (non-dimensional)")
    z_axis, r_axis = result.Z[0, :], result.R[:, 0]
    columns = [round(p * (len(z_axis) - 1)) for p in PROFILE_POSITIONS]

    for i, name in enumerate(result.outputs):
        values = result.fields[:, i].reshape(result.Z.shape)
        radial, axial = axes[i, 0], axes[i, 1]
        for column in columns:
            radial.plot(r_axis, values[:, column], label=f"z = {z_axis[column]:.3g}")
        radial.set_xlabel("r")
        radial.set_ylabel(name)
        radial.legend(fontsize=7)
        radial.grid(True, alpha=0.3)

        axial.plot(z_axis, values[0, :], label=f"axis (r = {r_axis[0]:.3g})")
        axial.plot(z_axis, values[-1, :], label=f"wall (r = {r_axis[-1]:.3g})")
        axial.set_xlabel("z")
        axial.set_ylabel(name)
        axial.legend(fontsize=7)
        axial.grid(True, alpha=0.3)
    axes[0, 0].set_title("Radial profiles")
    axes[0, 1].set_title("Axial profiles")
    fig.tight_layout()
    return save_figure(fig, output_dir / "profiles.png")


def plot_residual_maps(result: Evaluation, title: str, output_dir: Path) -> Path:
    rows = len(result.equations)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.4 * rows + 0.6), squeeze=False,
                             sharex=True, constrained_layout=True)
    fig.suptitle(f"{title}: absolute PDE residual")
    for i, (ax, name) in enumerate(zip(axes[:, 0], result.equations)):
        values = np.maximum(np.abs(result.grid_residuals[:, i]), 1e-20).reshape(result.Z.shape)
        mesh = ax.pcolormesh(result.Z, result.R, values, shading="auto", cmap="inferno",
                             norm=LogNorm(vmin=values.min(), vmax=values.max()))
        fig.colorbar(mesh, ax=ax, label=f"|{name}|")
        ax.set_ylabel("r")
    axes[-1, 0].set_xlabel("z")
    return save_figure(fig, output_dir / "residual_maps.png")


def plot_boundary_errors(result: Evaluation, title: str, output_dir: Path) -> Path:
    names = [name for name, metrics in result.bc_metrics.items() if metrics]
    rmse = [max(result.bc_metrics[name]["rmse"], 1e-16) for name in names]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(names) + 1.5))
    ax.barh(names, rmse, color="tab:blue")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (log scale), lower is better")
    ax.set_title(f"{title}: boundary-condition error")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, output_dir / "boundary_errors.png")


# ---------------------------------------------------------------------------
# Training-record plots
# ---------------------------------------------------------------------------
def mark_stages(ax, stages: list[dict[str, Any]]) -> None:
    """Vertical line and label at the start of each training stage."""
    for stage in stages:
        ax.axvline(stage["start_step"], color="gray", linewidth=0.8, linestyle="--")
        label = stage["name"] + (" (unweighted)" if stage["loss_weights"] is None else "")
        ax.text(stage["start_step"], 1.01, label, rotation=90, fontsize=7, va="bottom",
                transform=ax.get_xaxis_transform())


def plot_loss_history(record: dict[str, Any], term_names: list[str], title: str, output_dir: Path) -> Path:
    history = record["loss_history"]
    steps = np.array(history["steps"])
    losses = np.array(history["loss_train"])
    if len(term_names) != losses.shape[1]:
        term_names = [f"term {i}" for i in range(losses.shape[1])]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, max(losses.shape[1], 2)))
    for i, name in enumerate(term_names):
        ax.plot(steps, losses[:, i], color=colors[i], linewidth=1, label=name)
    ax.plot(steps, losses.sum(axis=1), color="black", linewidth=2, label="total")
    mark_stages(ax, record["stages"])
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss (as weighted during training)")
    ax.set_title(f"{title}: loss history", pad=70)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    return save_figure(fig, output_dir / "loss_history.png")


def plot_training_time(record: dict[str, Any], title: str, output_dir: Path) -> Path:
    stages = record["stages"]
    minutes = [stage["seconds"] / 60 for stage in stages]
    optimizers = sorted({stage["optimizer"] for stage in stages})
    colors = {name: plt.cm.tab10(i) for i, name in enumerate(optimizers)}

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(stages) + 1.8))
    ax.barh([stage["name"] for stage in stages], minutes,
            color=[colors[stage["optimizer"]] for stage in stages])
    for name, color in colors.items():
        ax.bar(0, 0, color=color, label=name)  # legend entry per optimizer
    ax.invert_yaxis()
    ax.set_xlabel("Wall time (minutes)")
    ax.set_title(f"{title}: training time — {record['training_seconds'] / 60:.1f} min training, "
                 f"{record['total_seconds'] / 60:.1f} min total")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, output_dir / "training_time.png")


def plot_rar_points(record: dict[str, Any], domain: list[list[float]], title: str, output_dir: Path) -> Path:
    rounds = record["rar_points"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(rounds)))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for color, item in zip(colors, rounds):
        points = np.array(item["points"])
        ax.scatter(points[:, 0], points[:, 1], s=4, color=color, label=f"round {item['round']}")
    (z_min, r_min), (z_max, r_max) = domain
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(r_min, r_max)
    ax.set_xlabel("z")
    ax.set_ylabel("r")
    ax.set_title(f"{title}: points added by residual-based adaptive refinement")
    ax.legend(fontsize=7, ncol=2, markerscale=3, loc="upper right")
    fig.tight_layout()
    return save_figure(fig, output_dir / "rar_points.png")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_metrics(model: pu.LoadedModel, result: Evaluation, output_dir: Path) -> Path:
    record = model.training_record
    metrics = {
        "model": model.info,
        "accuracy": {
            "training_objective": result.objective,
            "loss_weights": model.definitions["weights"],
            "pde_residuals": result.pde_metrics,
            "boundary_errors": result.bc_metrics,
        },
        "cross_section_averages": result.cross_section_averages,
        "training": None if record is None else {
            "training_seconds": record["training_seconds"],
            "total_seconds": record["total_seconds"],
            "device": record["device"],
            "gpu": record["gpu"],
            "stages": [{k: v for k, v in s.items() if k != "loss_weights"} for s in record["stages"]],
        },
    }
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def write_readme(model: pu.LoadedModel, result: Evaluation, output_dir: Path) -> Path:
    info, record = model.info, model.training_record
    lines = [
        f"# {Path(info['checkpoint']).name}",
        "",
        f"- Plotted: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- Physics: `{Path(info['script']).name}`",
        f"- Trained: {info['trained'] or 'unknown'} | Steps: {info['step']} | Layers: {info['layer_size']}",
    ]
    if record:
        lines.append(f"- Training time: {record['training_seconds'] / 60:.1f} min on {record['gpu'] or record['device']}")
    else:
        lines.append("- Training time: not recorded (no training record for this checkpoint)")
    if result.objective is not None:
        lines.append(f"- Training objective (weighted loss): {result.objective:.3e}")
    path = output_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def plot_model(checkpoint: str | Path, script: str | Path | None = None,
               output_directory: Path = OUTPUT_DIRECTORY) -> Path:
    """Evaluate one checkpoint and save all plots; returns the output folder."""
    model = pu.load_model(Path(checkpoint), pu.default_device(), script)
    title = model.info["version"] or Path(checkpoint).stem
    output_dir = Path(output_directory) / Path(model.info["checkpoint"]).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate(model)
    saved = [
        plot_fields(result, title, output_dir),
        plot_profiles(result, title, output_dir),
        plot_residual_maps(result, title, output_dir),
        plot_boundary_errors(result, title, output_dir),
    ]
    record = model.training_record
    if record:
        saved.append(plot_loss_history(record, result.equations + model.definitions["bc_names"],
                                       title, output_dir))
        saved.append(plot_training_time(record, title, output_dir))
        if record["rar_points"]:
            saved.append(plot_rar_points(record, model.info["domain"], title, output_dir))
    saved += [write_metrics(model, result, output_dir), write_readme(model, result, output_dir)]

    print(f"Plots saved to: {output_dir.resolve()}")
    for path in saved:
        print(f"  {path.name}")
    return output_dir


def plot_after_training(checkpoint: str | Path) -> None:
    """Called at the end of a training script; a plotting error never loses the run."""
    try:
        plot_model(checkpoint)
    except Exception as error:
        print(f"Plotting failed ({error}). The model is saved; rerun plot_pinn_model.py "
              f"with MODEL_PATH = '{checkpoint}' to retry.")


def main() -> None:
    checkpoint = pu.resolve_checkpoint(MODEL_PATH)
    if checkpoint is None:
        print(f"Checkpoint not found: {MODEL_PATH}\n\nAvailable checkpoints in models/<version>/ (oldest first):")
        for path in pu.available_checkpoints():
            print(f"  {path}")
        sys.exit(1)
    try:
        plot_model(checkpoint, MODEL_SCRIPT)
    except pu.PinnModelError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
