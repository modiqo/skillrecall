from __future__ import annotations

import io
import time

from skillrecall.progress import FRAMES, Ticker
from skillrecall.remote import guidance


def test_ticker_draws_frames_and_clears_when_enabled():
    buf = io.StringIO()
    t = Ticker(enabled=True, stream=buf).start("searching")
    time.sleep(0.2)
    t.stage("routing")
    time.sleep(0.1)
    t.progress(2, 5, "finished a")
    time.sleep(0.1)
    t.stop()
    out = buf.getvalue()
    assert any(f in out for f in FRAMES)
    assert "routing" in out and "2/5" in out
    assert out.endswith("\r")  # the line was cleared


def test_ticker_is_silent_when_disabled():
    buf = io.StringIO()
    t = Ticker(enabled=False, stream=buf).start("x")
    t.stage("y")
    t.stop()
    assert buf.getvalue() == ""


def test_guidance_lists_skills_and_commands():
    text = guidance("o/r", ["alpha", "beta", "gamma"])
    assert "o/r holds 3 skills" in text
    assert "skillrecall assess o/r/alpha" in text
    assert "skillrecall assess o/r\n" in text or text.rstrip().endswith("gamma")
    assert "alpha" in text and "gamma" in text


def test_guidance_suggests_close_match():
    text = guidance("o/r", ["code-review", "codebase-design"], missing="code-reveiw")
    assert "Did you mean: code-review" in text
    assert "skillrecall assess o/r/code-review" in text
