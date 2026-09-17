from pathlib import Path
import sys


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import query


def test_pipeline_attempt_summary_output_remains_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(
        query,
        "get_summary",
        lambda *args, **kwargs: {
            "counts": {"WAITING_QC": 2, "FAIL": 1},
            "failure_reasons": {"task_failed: check_accuracy": 1},
        },
    )

    query.show_summary(
        "dataset",
        pipeline_type=query.RECON_PIPELINE,
        latest_only=False,
    )

    output = capsys.readouterr().out
    assert (
        "=== Summary for dataset "
        "(mv_hoi_reconstruction, all rows) ==="
    ) in output
    assert "Total pipeline rows: 3" in output
    assert "  WAITING_QC: 2" in output
    assert "  FAIL: 1" in output
    assert "[1] task_failed: check_accuracy" in output
