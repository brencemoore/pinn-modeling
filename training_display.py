import sys
import time
import deepxde as dde


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
    """

    def __init__(self, optimizer, total_iterations=None, stage="",
                 display_every=1000, refresh_seconds=0.1):
        super().__init__()

        self.optimizer = optimizer
        self.total_iterations = total_iterations
        self.stage = stage
        # Must match the display_every passed to model.train() (DeepXDE default 1000)
        self.display_every = display_every
        self.refresh_seconds = refresh_seconds

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

    def _iteration(self):
        # Iteration relative to the beginning of THIS training session
        return self.model.train_state.step - self.start_iteration

    def _deepxde_will_print(self):
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
