"""Reject sequences where dummy_agent did not complete a clean render.

The sentinel file (`.dummy_agent_ok`) is written by
``scripts/rsl_rl/dummy_agent.py`` only after (a) the recording loop
finishes end-to-end, (b) ``env.close()`` returns — which is where
``gymnasium.wrappers.RecordVideo`` actually flushes the MP4 via moviepy,
because its in-loop check is a strict ``len(recorded_frames) >
video_length`` that never fires when the caller runs exactly
``video_length`` steps — and (c) the rendered MP4 is on disk with
non-zero size.  CUDA assert, uncaught exception, Omniverse-shutdown
deadlock, silent moviepy failure, or ``timeout --signal=KILL`` all
prevent the touch, so sentinel *presence* is a strict proof the sim
rendered the whole sequence AND the MP4 was persisted.
"""

from __future__ import annotations

from pathlib import Path

MARKER = ".dummy_agent_ok"


def check(data: dict, seq_dir: Path | None = None) -> dict:
    """Pass iff ``<seq_dir>/.dummy_agent_ok`` exists.

    ``seq_dir`` is the ``robot_name=<robot>`` partition dir — exactly where
    the workflow writes the sentinel (see ``workflow/retarget.yaml`` Stage 5).
    When running the assessor standalone without ``seq_dir`` wired in, this
    check passes vacuously so it doesn't block non-workflow invocations.
    """
    if seq_dir is None:
        return {"pass": True, "score": 1.0, "reason": "no seq_dir provided"}
    found = (Path(seq_dir) / MARKER).exists()
    return {
        "pass": found,
        "score": float(found),
        "reason": "sentinel present" if found else f"missing {MARKER}",
    }
