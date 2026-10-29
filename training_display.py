import shutil
import sys
import time
from pathlib import Path

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch


def _supports(text):
    # Some Windows consoles (cp1252) cannot print arrow characters
    try:
        text.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


ARROW_DOWN, ARROW_UP = ("↓", "↑") if _supports("↓↑") else ("v", "^")


class _LBFGSWatcher:
    """Reports L-BFGS progress from inside a DeepXDE L-BFGS chunk.

    DeepXDE runs PyTorch L-BFGS in chunks of up to 1000 iterations and only
    returns to the callbacks (and records the loss) after each chunk, which can
    take many minutes. Every L-BFGS loss evaluation calls
    model.data.losses_train, so this wraps that method: it copies the loss
    values and returns DeepXDE's result unchanged, so training is unaffected.

    Listeners are called with (iteration, weighted loss terms) once per
    finished L-BFGS iteration, using the lowest loss evaluated during that
    iteration's line search (normally the accepted point).
    """

    def __init__(self, model):
        self.model = model
        self.listeners = []
        self.iteration = None
        self.best = None

        original = model.data.losses_train

        def losses_train(*args, **kwargs):
            losses = original(*args, **kwargs)
            if self.listeners:
                self._record(losses)
            return losses

        model.data.losses_train = losses_train

    @classmethod
    def of(cls, model):
        """The watcher for this model, created on first use."""
        if not hasattr(model, "_lbfgs_watcher"):
            model._lbfgs_watcher = cls(model)
        return model._lbfgs_watcher

    def start(self, listener):
        self.listeners.append(listener)
        self.iteration = None
        self.best = None

    def stop(self, listener):
        if listener in self.listeners:
            self.listeners.remove(listener)

    def _record(self, losses):
        state = next(iter(self.model.opt.state.values()), {})
        iteration = int(state.get("n_iter", 0))

        terms = losses if isinstance(losses, list) else [losses]
        values = torch.stack(terms).detach().cpu().numpy().astype(float)
        if self.model.loss_weights is not None:
            values = values * np.asarray(self.model.loss_weights, dtype=float)

        if iteration != self.iteration:
            # The previous iteration is finished: report its accepted loss
            if self.best is not None:
                for listener in list(self.listeners):
                    listener(self.iteration, self.best)
            self.iteration, self.best = iteration, values
        elif values.sum() < self.best.sum():
            self.best = values


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

    With deepxde_prints=False the line also updates during L-BFGS chunks,
    showing the L-BFGS iteration and its loss (see _LBFGSWatcher).
    """

    def __init__(self, optimizer, total_iterations=None, stage="",
                 display_every=1000, refresh_seconds=0.1,
                 num_pde_terms=None, deepxde_prints=True, log_seconds=30):
        super().__init__()

        # In a terminal the line updates in place; otherwise (log file, pipe)
        # a full line is printed every log_seconds
        self.interactive = sys.stdout.isatty()
        self.log_seconds = log_seconds
        self.last_log = 0.0
        self.last_logged_iteration = None

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
        self.live_iteration = 0
        self.last_draw = 0.0
        self.line_length = 0

        self.loss_source = None
        self.current_loss = None
        self.loss_step = None
        self.arrow = ""

    def on_train_begin(self):
        self.start_time = time.time()
        self.last_draw = 0.0
        self.last_log = 0.0
        self.last_logged_iteration = None
        self.line_length = 0

        # Remember where this training session started
        self.start_iteration = self.model.train_state.step
        self.live_iteration = 0

        # DeepXDE evaluates the loss once before training starts
        self.loss_source = None
        self.current_loss = None
        self.arrow = ""
        self._update_loss()

        # Mid-chunk updates would collide with DeepXDE's own table rows
        if self._is_lbfgs() and not self.deepxde_prints:
            _LBFGSWatcher.of(self.model).start(self._on_lbfgs_iteration)

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
        _LBFGSWatcher.of(self.model).stop(self._on_lbfgs_iteration)

        # Leave the final progress line in place for this stage
        self._update_loss()
        if self.last_logged_iteration != self._iteration():
            self.last_log = 0.0  # always log the final line when not in a terminal
        self._draw(time.time())
        if self.interactive:
            print()
        if self.num_pde_terms is not None and self.loss_source is not None:
            loss = np.asarray(self.loss_source, dtype=float)
            pde = loss[:self.num_pde_terms].sum()
            bc = loss[self.num_pde_terms:].sum()
            print(f"{'':<6}   {'':<10}   PDE loss {pde:.2e} | BC loss {bc:.2e}")

    def _on_lbfgs_iteration(self, iteration, losses):
        self.live_iteration = iteration
        self._set_loss(float(losses.sum()), self.start_iteration + iteration)
        now = time.time()
        if now - self.last_draw >= self.refresh_seconds:
            self._draw(now)

    def _is_lbfgs(self):
        return self.model.opt_name.startswith("L-BFGS")

    def _iteration(self):
        # Iteration relative to the beginning of THIS training session
        return max(self.model.train_state.step - self.start_iteration, self.live_iteration)

    def _deepxde_will_print(self):
        if not self.deepxde_prints:
            return False
        if self._is_lbfgs():
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
        self.loss_source = loss
        self._set_loss(float(sum(loss)), self.model.train_state.step)

    def _set_loss(self, new_loss, step):
        # Changes under 0.1% (e.g. DeepXDE's recheck of the last L-BFGS loss) keep the arrow
        if self.current_loss is None:
            self.arrow = ""
        elif new_loss < self.current_loss * 0.999:
            self.arrow = ARROW_DOWN
        elif new_loss > self.current_loss * 1.001:
            self.arrow = ARROW_UP
        self.current_loss = new_loss
        self.loss_step = step

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

        if not self.interactive:
            # Not a terminal (log file, pipe): \r is not understood, so print
            # a full line every log_seconds and when the stage finishes
            finished = iteration == self.total_iterations
            if now - self.last_log >= self.log_seconds or finished or self.last_log == 0.0:
                if iteration != self.last_logged_iteration or self.last_log == 0.0:
                    print(line, flush=True)
                self.last_log = now
                self.last_logged_iteration = iteration
            return

        # A line wider than the terminal wraps, and \r then only returns to its
        # last row, so every update would pile up: shorten it to fit instead
        width = shutil.get_terminal_size(fallback=(120, 24)).columns - 1
        if len(line) > width:
            line = (
                f"{self.optimizer} {self.stage} | "
                f"{progress} {percent_text.strip()} | "
                f"{loss_text.replace(' (step ', ' (').replace('Loss ', '')} | "
                f"{self._format_time(elapsed)} {eta_text}"
            ).rstrip()[:width]

        # Pad with spaces so a shorter line fully overwrites the previous one
        padding = max(min(self.line_length, width) - len(line), 0)
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
    L-BFGS chunk of up to 1000 iterations. Inside an L-BFGS chunk they also
    redraw every `lbfgs_redraw_seconds` with the loss of each L-BFGS iteration
    (see _LBFGSWatcher). Window events are handled between redraws so the
    windows can be moved and resized while training runs.
    """

    FILE_NAMES = ("live_loss_summary.png", "live_loss_terms.png")

    def __init__(self, term_names, num_pde_terms, title="Training loss",
                 lbfgs_redraw_seconds=10.0, event_seconds=0.1):
        super().__init__()
        self.term_names = list(term_names)
        self.num_pde_terms = num_pde_terms
        self.title = title
        self.lbfgs_redraw_seconds = lbfgs_redraw_seconds
        self.event_seconds = event_seconds

        self.next_stage = None
        self.stage_starts = []  # (step, name) for each model.train() call
        self.points_drawn = 0
        self.live_steps = []    # L-BFGS iterations not yet recorded by DeepXDE
        self.live_losses = []
        self.last_redraw = 0.0
        self.last_events = 0.0
        self.laid_out = []      # figures whose margins are already set
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
        self.live_steps, self.live_losses = [], []
        if self.model.opt_name.startswith("L-BFGS"):
            _LBFGSWatcher.of(self.model).start(self._on_lbfgs_iteration)
        self._redraw(force=True)

    def on_epoch_end(self):
        self._redraw()
        self._process_events()

    def on_train_end(self):
        _LBFGSWatcher.of(self.model).stop(self._on_lbfgs_iteration)
        self.live_steps, self.live_losses = [], []
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

    def _on_lbfgs_iteration(self, iteration, losses):
        self.live_steps.append(self.stage_starts[-1][0] + iteration)
        self.live_losses.append(losses)
        if time.time() - self.last_redraw >= self.lbfgs_redraw_seconds:
            self._redraw(force=True)
        else:
            self._process_events()

    def _process_events(self):
        # Lets the windows respond to moving and resizing between redraws
        now = time.time()
        if now - self.last_events < self.event_seconds:
            return
        self.last_events = now
        for fig in (self.summary_fig, self.terms_fig):
            if fig is not None:
                fig.canvas.flush_events()

    def _redraw(self, force=False):
        # losshistory gets a new row only when DeepXDE recomputes the loss
        history = self.model.losshistory
        if not history.steps or (not force and len(history.steps) == self.points_drawn):
            return
        self.points_drawn = len(history.steps)
        self.last_redraw = time.time()

        steps = np.array(history.steps)
        losses = np.vstack(history.loss_train)  # already multiplied by loss_weights

        # Add L-BFGS iterations from inside the current chunk
        live = [i for i, step in enumerate(self.live_steps) if step > steps[-1]]
        if live:
            steps = np.concatenate([steps, np.array(self.live_steps)[live]])
            losses = np.vstack([losses, np.array(self.live_losses)[live]])

        names = self.term_names if len(self.term_names) == losses.shape[1] else [
            f"term {i}" for i in range(losses.shape[1])]

        ax = self.summary_fig.axes[0]
        ax.clear()
        ax.plot(steps, losses.sum(axis=1), color="black", linewidth=2, label="total")
        ax.plot(steps, losses[:, :self.num_pde_terms].sum(axis=1), linewidth=1.2, label="PDE")
        ax.plot(steps, losses[:, self.num_pde_terms:].sum(axis=1), linewidth=1.2, label="boundary conditions")
        self._format(ax, "summary", steps[-1], fontsize=9)

        ax = self.terms_fig.axes[0]
        ax.clear()
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(names), 2)))
        for i, name in enumerate(names):
            style = "-" if i < self.num_pde_terms else "--"
            ax.plot(steps, losses[:, i], style, color=colors[i], linewidth=1, label=name)
        self._format(ax, "every term", steps[-1], fontsize=7, ncol=2)

        for fig in (self.summary_fig, self.terms_fig):
            # tight_layout is slow (~0.4 s per figure), so the margins are set once
            if fig not in self.laid_out:
                fig.tight_layout()
                self.laid_out.append(fig)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        self.last_events = time.time()

    def _format(self, ax, kind, step, fontsize, ncol=1):
        for start, name in self.stage_starts:
            ax.axvline(start, color="gray", linewidth=0.8, linestyle="--")
            ax.text(start, 1.01, name, rotation=90, fontsize=7, va="bottom",
                    transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Training loss (as weighted during training)")
        ax.set_title(f"{self.title}: {kind} (step {step})", pad=55)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=fontsize, ncol=ncol, loc="upper right")
