"""A small progress ticker for the terminal.

Renders on stderr only when stderr is a terminal, so piped or captured
output never sees it. A background thread redraws the line a few times a
second; the work itself runs on the calling thread, which just updates the
stage label. Braille dot-matrix frames keep the spinner narrow and legible
in any monospace font.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
BAR_FULL, BAR_EMPTY = "⣿", "⣀"
INTERVAL = 0.08


class Ticker:
    def __init__(self, enabled: bool | None = None, stream=None) -> None:
        self.stream = stream or sys.stderr
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self._lock = threading.Lock()
        self._stage = ""
        self._done = 0
        self._total = 0
        self._current = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._width = 0

    # -- control -----------------------------------------------------------

    def start(self, stage: str = "starting") -> Ticker:
        self._stage = stage
        self._started = time.monotonic()
        if self.enabled and self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="skillrecall-ticker", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.enabled:
            self._clear()

    def __enter__(self) -> Ticker:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- updates from the working thread -------------------------------------

    def stage(self, text: str) -> None:
        with self._lock:
            self._stage = text

    def progress(self, done: int, total: int, current: str = "") -> None:
        with self._lock:
            self._done, self._total, self._current = done, total, current

    def stage_callback(self) -> Callable[[str], None]:
        return self.stage

    # -- rendering -----------------------------------------------------------

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            self._draw(FRAMES[i % len(FRAMES)])
            i += 1
            self._stop.wait(INTERVAL)

    def _draw(self, frame: str) -> None:
        with self._lock:
            stage, done, total, current = self._stage, self._done, self._total, self._current
        elapsed = time.monotonic() - self._started
        if total:
            width = 20
            filled = int(width * done / total)
            bar = BAR_FULL * filled + BAR_EMPTY * (width - filled)
            line = f"{frame} {bar} {done}/{total} {elapsed:4.0f}s  {current}"
        else:
            line = f"{frame} {stage}  {elapsed:4.0f}s"
        line = line[:120]
        pad = max(0, self._width - len(line))
        self._width = len(line)
        try:
            self.stream.write(f"\r{line}{' ' * pad}")
            self.stream.flush()
        except (OSError, ValueError):  # closed stream
            self.enabled = False
            self._stop.set()

    def _clear(self) -> None:
        try:
            self.stream.write("\r" + " " * self._width + "\r")
            self.stream.flush()
        except (OSError, ValueError):
            pass
