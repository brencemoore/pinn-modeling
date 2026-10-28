import sys
import time
from pathlib import Path

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np


def _supports(text):
    # Some Windows consoles (cp1252) cannot print arrow characters
    try:
        text.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


ARROW_DOWN, ARROW_UP = ("↓", "↑") if _supports("↓↑") else ("v", "^")


class TrainingProgress(dde.callbacks.Callback):
    """Single-line live progress display for model.train().

    Read-only: it never changes the model, data, or random state, so training
    results are identical with or without it.

    DeepXDE only recomputes the training loss when it prints a row of its loss
    table (every `display_every` steps for Adam, every L-BFGS chunk), so the
    loss shown here is the most recent table value and the step it came from.

    For a cleaner terminal, call model.train(..., verbose=0) to hide DeepXDE's
    loss table and pass deepxde_prints=False. With num_pde_terms set, a short
    PDE / boundary-condition loss split is printed when each stage ends.
    """

    def __init__(self, optimizer, total_iterations=None, stage="",
                 display_every=1000, refresh_seconds=0.1,
                 num_pde_terms=None, deepxde_prints=True):
        super().__init__()

        self.optimizer = optimizer
        self.total_iterations = total_iterations
        self.stage = stage
        # Must match the display_every passed to model.train() (DeepXDE default 1000)
        self.display_every = display_every
        self.refresh_seconds = refresh_seconds
        self.num_pde_terms = num_pde_terms
        # Must be False when model.train() is called with verbose=0
        self.deepxde_prints = deepxde_prints

        self.start_time = None
        self.start_iteration = None
        self.last_draw = 0.0
        self.line_length = 0

        self.loss_source = None
        self.current_loss = None
        self.loss_step = None
        self.arrow = ""

    def on_train_begin(self):
        self.start_time = time.time()
        self.last_draw = 0.0
        self.line_length = 0

        # Remember where this training session started
        self.start_iteration = self.model.train_state.step

        # DeepXDE evaluates the loss once before training starts
        self.loss_source = None
        self.current_loss = None
        self.arrow = ""
        self._update_loss()

    def on_epoch_begin(self):
        # Clear the progress line before DeepXDE prints a loss table row,
        # otherwise the row is appended to the end of the progress text
        if self._deepxde_will_print():
            self._clear_line()

    def on_epoch_end(self):
        self._update_loss()

        now = time.time()
        finished = (self.total_iterations is not None
                    and self._iteration() >= self.total_iterations)
        if now - self.last_draw >= self.refresh_seconds or finished:
            self._draw(now)

    def on_train_end(self):
        # Leave the final progress line in place for this stage
        self._draw(time.time())
        print()
        if self.num_pde_terms is not None and self.loss_source is not None:
            loss = np.asarray(self.loss_source, dtype=float)
            pde = loss[:self.num_pde_terms].sum()
            bc = loss[self.num_pde_terms:].sum()
            print(f"{'':<6}   {'':<10}   PDE loss {pde:.2e} | BC loss {bc:.2e}")

    def _iteration(self):
        # Iteration relative to the beginning of THIS training session
        return self.model.train_state.step - self.start_iteration

    def _deepxde_will_print(self):
        if not self.deepxde_prints:
            return False
        if self.model.opt_name.startswith("L-BFGS"):
            # PyTorch L-BFGS prints a row after every chunk of iterations
            return True
        next_iteration = self._iteration() + 1
        return ((self.model.train_state.step + 1) % self.display_every == 0
                or next_iteration == self.total_iterations)

    def _update_loss(self):
        # DeepXDE replaces loss_train with a new array each time it recomputes it
        loss = self.model.train_state.loss_train
        if loss is None or loss is self.loss_source:
            return

        new_loss = float(sum(loss))
        if self.current_loss is None:
            self.arrow = ""
        elif new_loss < self.current_loss:
            self.arrow = ARROW_DOWN
        else:
            self.arrow = ARROW_UP

        self.loss_source = loss
        self.current_loss = new_loss
        self.loss_step = self.model.train_state.step

    def _draw(self, now):
        self.last_draw = now
        iteration = self._iteration()
        elapsed = now - self.start_time

        # Progress, percentage, and estimated time remaining
        if self.total_iterations:
            percent = min(iteration / self.total_iterations, 1.0) * 100
            progress = f"{iteration}/{self.total_iterations}"
            percent_text = f"{percent:5.1f}%"
            if iteration > 0:
                remaining = elapsed / iteration * max(self.total_iterations - iteration, 0)
                eta_text = f"ETA {self._format_time(remaining)}"
            else:
                eta_text = "ETA --:--"
        else:
            progress = str(iteration)
            percent_text = "   --"
            eta_text = ""

        if self.current_loss is None:
            loss_text = "Loss --"
        else:
            loss_text = f"Loss {self.current_loss:.2e} {self.arrow:1} (step {self.loss_step})"

        line = (
            f"{self.optimizer:<6} | "
            f"{self.stage:<10} | "
            f"Iteration {progress:<11} | "
            f"{percent_text} | "
            f"{loss_text} | "
            f"{self._format_time(elapsed)} {eta_text}"
        ).rstrip()

        # Pad with spaces so a shorter line fully overwrites the previous one
        padding = max(self.line_length - len(line), 0)
        print("\r" + line + " " * padding, end="", flush=True)
        self.line_length = len(line)

    def _clear_line(self):
        if self.line_length:
            print("\r" + " " * self.line_length + "\r", end="", flush=True)
            self.line_length = 0

    @staticmethod
    def _format_time(seconds):
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


class LiveLossPlot(dde.callbacks.Callback):
    """Live loss plots for model.train(), in two windows:

      summary   total, PDE, and boundary-condition loss
      terms     every loss term

    Use one instance for every model.train() call so the plots cover the whole
    run, and call stage(name) in the callbacks list to label each call. Like
    TrainingProgress it is read-only, so training results are unchanged.

    The plots redraw whenever DeepXDE records a new loss: every `display_every`
    steps for Adam (pass display_every to model.train()), and after every
    L-BFGS chunk of up to 1000 iterations.
    """

    FILE_NAMES = ("live_loss_summary.png", "live_loss_terms.png")

    def __init__(self, term_names, num_pde_terms, title="Training loss"):
        super().__init__()
        self.term_names = list(term_names)
        self.num_pde_terms = num_pde_terms
        self.title = title

        self.next_stage = None
        self.stage_starts = []  # (step, name) for each model.train() call
        self.points_drawn = 0
        self.summary_fig = self.terms_fig = None

    def stage(self, name):
        """Label the next model.train() call; returns self for the callbacks list."""
        self.next_stage = name
        return self

    def on_train_begin(self):
        if self.summary_fig is None:
            self.summary_fig, _ = plt.subplots(figsize=(9, 5))
            self.terms_fig, _ = plt.subplots(figsize=(11, 6))
            plt.show(block=False)
        name = self.next_stage or self.model.opt_name
        self.stage_starts.append((self.model.train_state.step, name))
        self.next_stage = None
        self._redraw(force=True)

    def on_epoch_end(self):
        self._redraw()

    def on_train_end(self):
        self._redraw(force=True)

    def save(self, output_dir):
        """Save both plots to output_dir, close their windows, and return the paths."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for fig, name in zip((self.summary_fig, self.terms_fig), self.FILE_NAMES):
            if fig is None:
                continue
            fig.savefig(output_dir / name, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths.append(output_dir / name)
        self.summary_fig = self.terms_fig = None
        return paths

    def _redraw(self, force=False):
        # losshistory gets a new row only when DeepXDE recomputes the loss
        history = self.model.losshistory
        if not history.steps or (not force and len(history.steps) == self.points_drawn):
            return
        self.points_drawn = len(history.steps)

        steps = np.array(history.steps)
        losses = np.vstack(history.loss_train)  # already multiplied by loss_weights
        names = self.term_names if len(self.term_names) == losses.shape[1] else [
            f"term {i}" for i in range(losses.shape[1])]

        ax = self.summary_fig.axes[0]
        ax.clear()
        ax.plot(steps, losses.sum(axis=1), color="black", linewidth=2, label="total")
        ax.plot(steps, losses[:, :self.num_pde_terms].sum(axis=1), linewidth=1.2, label="PDE")
        ax.plot(steps, losses[:, self.num_pde_terms:].sum(axis=1), linewidth=1.2, label="boundary conditions")
        self._format(ax, "summary", fontsize=9)

        ax = self.terms_fig.axes[0]
        ax.clear()
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(names), 2)))
        for i, name in enumerate(names):
            style = "-" if i < self.num_pde_terms else "--"
            ax.plot(steps, losses[:, i], style, color=colors[i], linewidth=1, label=name)
        self._format(ax, "every term", fontsize=7, ncol=2)

        for fig in (self.summary_fig, self.terms_fig):
            fig.tight_layout()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _format(self, ax, kind, fontsize, ncol=1):
        for step, name in self.stage_starts:
            ax.axvline(step, color="gray", linewidth=0.8, linestyle="--")
            ax.text(step, 1.01, name, rotation=90, fontsize=7, va="bottom",
                    transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Training loss (as weighted during training)")
        ax.set_title(f"{self.title}: {kind} (step {self.model.train_state.step})", pad=55)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=fontsize, ncol=ncol, loc="upper right")
