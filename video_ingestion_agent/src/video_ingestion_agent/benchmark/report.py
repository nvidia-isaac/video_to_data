#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Benchmark report generator for EPIC-KITCHENS-100 evaluation.

Generates an HTML report with:
  - Per-video timeline: predicted vs GT segments side-by-side
  - Aggregate metrics table: mAP@tIoU, boundary P/R, verb/noun accuracy
  - Error analysis: over/under-segmentation, confused verbs/nouns
  - Comparison placeholders for EPIC-KITCHENS leaderboard baselines

Usage:
    python -m video_ingestion_agent.benchmark.report \\
        --eval-results runs/benchmark_epic_kitchens/eval_results.json \\
        --predictions runs/benchmark_epic_kitchens/mapped_predictions.jsonl \\
        --annotations-dir data/benchmark/epic_kitchens/annotations \\
        --output runs/benchmark_epic_kitchens/benchmark_report.html
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EPIC-KITCHENS-100 Benchmark Report</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent2: #a78bfa;
            --success: #4ade80;
            --warning: #fbbf24;
            --danger: #f87171;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
            line-height: 1.6;
        }}
        h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; color: var(--accent); }}
        h2 {{ font-size: 1.3rem; margin: 1.5rem 0 0.75rem; color: var(--accent2); }}
        h3 {{ font-size: 1.1rem; margin: 1rem 0 0.5rem; color: var(--text); }}
        .header {{ margin-bottom: 2rem; }}
        .header p {{ color: var(--text-muted); font-size: 0.9rem; }}
        .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        th, td {{
            padding: 0.5rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ color: var(--accent); font-weight: 600; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        .metric-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1rem;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent);
        }}
        .metric-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .timeline {{
            position: relative;
            margin: 0.5rem 0;
            height: 24px;
            background: rgba(255,255,255,0.05);
            border-radius: 4px;
            overflow: hidden;
        }}
        .timeline-seg {{
            position: absolute;
            height: 100%;
            border-radius: 3px;
            opacity: 0.8;
            cursor: pointer;
            transition: opacity 0.2s;
            font-size: 0.6rem;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            overflow: hidden;
            white-space: nowrap;
        }}
        .timeline-seg:hover {{ opacity: 1; }}
        .timeline-gt {{ background: var(--success); }}
        .timeline-pred {{ background: var(--accent); }}
        .legend {{
            display: flex;
            gap: 1.5rem;
            margin: 0.5rem 0;
            font-size: 0.8rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}
        .good {{ color: var(--success); }}
        .warn {{ color: var(--warning); }}
        .bad {{ color: var(--danger); }}
        .tooltip {{
            position: relative;
        }}
        .video-section {{
            margin-bottom: 1.5rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>EPIC-KITCHENS-100 Benchmark Report</h1>
        <p>video_ingestion_agent segmentation pipeline evaluation</p>
        <p>Generated: {timestamp}</p>
    </div>

    {summary_section}

    {temporal_section}

    {annotation_section}

    {timeline_section}

    {error_analysis_section}

</body>
</html>"""


# ---------------------------------------------------------------------------
# Report generation functions
# ---------------------------------------------------------------------------


def _format_metric(value, fmt=".4f") -> str:
    """Format a metric value."""
    if isinstance(value, float):
        return f"{value:{fmt}}"
    return str(value)


def _color_class(value: float, thresholds=(0.3, 0.6)) -> str:
    """Return CSS class based on value quality."""
    if value >= thresholds[1]:
        return "good"
    elif value >= thresholds[0]:
        return "warn"
    return "bad"


def generate_summary_section(eval_results: dict, meta: dict) -> str:
    """Generate the summary metrics cards."""
    temporal = eval_results.get("temporal", {})
    annotation = eval_results.get("annotation", {})
    metadata = eval_results.get("metadata", {})

    map_avg = temporal.get("mAP", {}).get("mAP_avg", 0.0)
    map_05 = temporal.get("mAP", {}).get("mAP@0.5", 0.0)

    seg_ratio = temporal.get("segmentation_ratio", {}).get("mean_ratio", 0.0)
    verb_acc = annotation.get("accuracy", {}).get("verb_top1_acc", 0.0)
    noun_acc = annotation.get("accuracy", {}).get("noun_top1_acc", 0.0)

    sem_sim = annotation.get("semantic_similarity", {}).get("mean_similarity", 0.0)
    bert_f1 = annotation.get("bertscore", {}).get("f1", 0.0)

    n_videos = metadata.get("n_gt_videos", 0)
    n_pred = metadata.get("n_pred_segments", 0)
    n_gt = metadata.get("n_gt_segments", 0)

    return f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value {_color_class(map_avg)}">{map_avg:.3f}</div>
            <div class="metric-label">mAP (avg)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {_color_class(map_05)}">{map_05:.3f}</div>
            <div class="metric-label">mAP@0.5</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {_color_class(1.0 - abs(1.0 - seg_ratio))}">{seg_ratio:.2f}</div>
            <div class="metric-label">Seg Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {_color_class(verb_acc)}">{verb_acc:.3f}</div>
            <div class="metric-label">Verb Top-1</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {_color_class(noun_acc)}">{noun_acc:.3f}</div>
            <div class="metric-label">Noun Top-1</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {_color_class(sem_sim)}">{sem_sim:.3f}</div>
            <div class="metric-label">Semantic Sim</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {_color_class(bert_f1)}">{bert_f1:.3f}</div>
            <div class="metric-label">BERT F1</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{n_videos}</div>
            <div class="metric-label">Videos</div>
        </div>
    </div>
    <div class="card">
        <p>Predictions: {n_pred} segments | Ground Truth: {n_gt} segments</p>
    </div>
    """


def generate_temporal_section(eval_results: dict) -> str:
    """Generate the temporal metrics section."""
    temporal = eval_results.get("temporal", {})

    # mAP table
    map_data = temporal.get("mAP", {})
    map_rows = ""
    for key in sorted(map_data.keys()):
        val = map_data[key]
        if not isinstance(val, (int, float)):
            continue  # skip metadata keys like _eval_source
        css = _color_class(val)
        map_rows += f'<tr><td>{key}</td><td class="{css}">{val:.4f}</td></tr>\n'

    # Boundary table
    boundary_data = temporal.get("boundary", {})
    boundary_rows = ""
    for key, metrics in boundary_data.items():
        if isinstance(metrics, dict):
            p = metrics.get("precision", 0.0)
            r = metrics.get("recall", 0.0)
            f1 = metrics.get("f1", 0.0)
            boundary_rows += (
                f"<tr><td>{key}</td>"
                f'<td class="{_color_class(p)}">{p:.4f}</td>'
                f'<td class="{_color_class(r)}">{r:.4f}</td>'
                f'<td class="{_color_class(f1)}">{f1:.4f}</td></tr>\n'
            )

    # Segmentation ratio
    seg_ratio = temporal.get("segmentation_ratio", {})
    seg_info = (
        f"Mean: {seg_ratio.get('mean_ratio', 0):.2f} | "
        f"Median: {seg_ratio.get('median_ratio', 0):.2f} | "
        f"Over-seg: {seg_ratio.get('over_segmented', 0)} | "
        f"Under-seg: {seg_ratio.get('under_segmented', 0)} | "
        f"Well-seg: {seg_ratio.get('well_segmented', 0)}"
    )

    return f"""
    <h2>Temporal Segmentation Metrics</h2>
    <div class="card">
        <h3>mAP@tIoU</h3>
        <table>
            <tr><th>Threshold</th><th>mAP</th></tr>
            {map_rows}
        </table>
    </div>
    <div class="card">
        <h3>Boundary Precision / Recall</h3>
        <table>
            <tr><th>Tolerance</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
            {boundary_rows}
        </table>
    </div>
    <div class="card">
        <h3>Segmentation Ratio</h3>
        <p>{seg_info}</p>
    </div>
    """


def generate_annotation_section(eval_results: dict) -> str:
    """Generate the annotation accuracy section."""
    annotation = eval_results.get("annotation", {})

    # Accuracy
    acc = annotation.get("accuracy", {})
    acc_rows = ""
    for key in ["verb_top1_acc", "noun_top1_acc", "verb_top5_acc", "noun_top5_acc"]:
        val = acc.get(key, 0.0)
        css = _color_class(val)
        acc_rows += f'<tr><td>{key}</td><td class="{css}">{val:.4f}</td></tr>\n'

    # Soft match
    soft = annotation.get("soft_match", {})
    soft_rows = ""
    for key in ["verb_soft_match_rate", "noun_soft_match_rate"]:
        val = soft.get(key, 0.0)
        css = _color_class(val)
        soft_rows += f'<tr><td>{key}</td><td class="{css}">{val:.4f}</td></tr>\n'

    # Semantic similarity
    sem = annotation.get("semantic_similarity", {})
    sem_rows = ""
    for key in ["mean_similarity", "median_similarity", "std_similarity"]:
        val = sem.get(key, 0.0)
        if isinstance(val, float):
            sem_rows += f"<tr><td>{key}</td><td>{val:.4f}</td></tr>\n"

    # BERTScore
    bert = annotation.get("bertscore", {})
    bert_rows = ""
    for key in ["precision", "recall", "f1"]:
        val = bert.get(key, 0.0)
        if isinstance(val, float):
            css = _color_class(val)
            bert_rows += f'<tr><td>BERTScore {key}</td><td class="{css}">{val:.4f}</td></tr>\n'

    return f"""
    <h2>Annotation Accuracy Metrics</h2>
    <div class="card">
        <h3>Classification Accuracy (after adapter mapping)</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {acc_rows}
        </table>
    </div>
    <div class="card">
        <h3>Soft Match Rate (substring match)</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {soft_rows}
        </table>
    </div>
    <div class="card">
        <h3>Semantic Similarity (sentence-transformers)</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {sem_rows}
        </table>
    </div>
    <div class="card">
        <h3>BERTScore (narration quality)</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {bert_rows}
        </table>
    </div>
    """


def generate_timeline_section(
    eval_results: dict,
    predictions: dict[str, list] | None = None,
    gt_segments: dict[str, list] | None = None,
) -> str:
    """Generate per-video timeline visualization."""
    per_video = eval_results.get("per_video", {})

    if not per_video:
        return "<h2>Per-Video Timelines</h2><p>No per-video data available.</p>"

    timeline_html = "<h2>Per-Video Timelines</h2>\n"
    timeline_html += """
    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background: var(--success);"></div> Ground Truth</div>
        <div class="legend-item"><div class="legend-dot" style="background: var(--accent);"></div> Prediction</div>
    </div>
    """

    for video_id in sorted(per_video.keys()):
        info = per_video[video_id]
        n_pred = info.get("n_pred", 0)
        n_gt = info.get("n_gt", 0)
        ratio = info.get("seg_ratio", 0.0)
        map3 = info.get("mAP@0.3", 0.0)
        map5 = info.get("mAP@0.5", 0.0)

        # Generate timelines if we have segment data
        gt_timeline = ""
        pred_timeline = ""

        if gt_segments and video_id in gt_segments:
            gt_segs = gt_segments[video_id]
            if gt_segs:
                max_t = max(s["end_t"] for s in gt_segs) if gt_segs else 1.0
                max_t = max(max_t, 1.0)
                for seg in gt_segs:
                    left_pct = (seg["start_t"] / max_t) * 100
                    width_pct = max(((seg["end_t"] - seg["start_t"]) / max_t) * 100, 0.5)
                    label = seg.get("narration", seg.get("verb", ""))[:15]
                    gt_timeline += (
                        f'<div class="timeline-seg timeline-gt" '
                        f'style="left:{left_pct:.1f}%;width:{width_pct:.1f}%" '
                        f'title="{seg.get("narration", "")}">{label}</div>'
                    )

        if predictions and video_id in predictions:
            pred_segs = predictions[video_id]
            if pred_segs:
                max_t_pred = max(s["end_t"] for s in pred_segs) if pred_segs else 1.0
                # Use same max_t as GT if available
                if gt_segments and video_id in gt_segments and gt_segments[video_id]:
                    max_t_pred = max(max_t_pred, max(s["end_t"] for s in gt_segments[video_id]))
                max_t_pred = max(max_t_pred, 1.0)
                for seg in pred_segs:
                    left_pct = (seg["start_t"] / max_t_pred) * 100
                    width_pct = max(((seg["end_t"] - seg["start_t"]) / max_t_pred) * 100, 0.5)
                    label = seg.get("raw_description", seg.get("description", ""))[:15]
                    pred_timeline += (
                        f'<div class="timeline-seg timeline-pred" '
                        f'style="left:{left_pct:.1f}%;width:{width_pct:.1f}%" '
                        f'title="{seg.get("raw_description", seg.get("description", ""))}">'
                        f"{label}</div>"
                    )

        ratio_class = _color_class(1.0 - abs(1.0 - ratio))

        timeline_html += f"""
        <div class="card video-section">
            <h3>{video_id}</h3>
            <p>Pred: {n_pred} | GT: {n_gt} |
               Ratio: <span class="{ratio_class}">{ratio:.2f}</span> |
               mAP@0.3: {map3:.3f} | mAP@0.5: {map5:.3f}</p>
            <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.3rem;">GT:</div>
            <div class="timeline">{gt_timeline}</div>
            <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.3rem;">Pred:</div>
            <div class="timeline">{pred_timeline}</div>
        </div>
        """

    return timeline_html


def generate_error_analysis_section(eval_results: dict) -> str:
    """Generate error analysis section."""
    temporal = eval_results.get("temporal", {})
    per_video = eval_results.get("per_video", {})

    # Find worst/best videos
    if per_video:
        sorted_by_map = sorted(
            per_video.items(),
            key=lambda x: x[1].get("mAP@0.3", 0.0),
        )
        worst_5 = sorted_by_map[:5]
        best_5 = sorted_by_map[-5:][::-1]
    else:
        worst_5 = best_5 = []

    worst_rows = ""
    for vid, info in worst_5:
        worst_rows += (
            f"<tr><td>{vid}</td>"
            f"<td>{info.get('mAP@0.3', 0):.3f}</td>"
            f"<td>{info.get('seg_ratio', 0):.2f}</td>"
            f"<td>{info.get('n_pred', 0)}/{info.get('n_gt', 0)}</td></tr>\n"
        )

    best_rows = ""
    for vid, info in best_5:
        best_rows += (
            f"<tr><td>{vid}</td>"
            f"<td>{info.get('mAP@0.3', 0):.3f}</td>"
            f"<td>{info.get('seg_ratio', 0):.2f}</td>"
            f"<td>{info.get('n_pred', 0)}/{info.get('n_gt', 0)}</td></tr>\n"
        )

    # Segmentation ratio analysis
    seg_ratio = temporal.get("segmentation_ratio", {})
    over_seg = seg_ratio.get("over_segmented", 0)
    under_seg = seg_ratio.get("under_segmented", 0)
    well_seg = seg_ratio.get("well_segmented", 0)
    total = over_seg + under_seg + well_seg

    seg_analysis = ""
    if total > 0:
        seg_analysis = f"""
        <p>Segmentation distribution:
            <span class="good">{well_seg} well-segmented ({well_seg / total * 100:.0f}%)</span> |
            <span class="warn">{over_seg} over-segmented ({over_seg / total * 100:.0f}%)</span> |
            <span class="bad">{under_seg} under-segmented ({under_seg / total * 100:.0f}%)</span>
        </p>
        """

    return f"""
    <h2>Error Analysis</h2>
    <div class="card">
        <h3>Segmentation Pattern</h3>
        {seg_analysis}
    </div>
    <div class="card">
        <h3>Best Performing Videos</h3>
        <table>
            <tr><th>Video</th><th>mAP@0.3</th><th>Seg Ratio</th><th>Pred/GT</th></tr>
            {best_rows}
        </table>
    </div>
    <div class="card">
        <h3>Worst Performing Videos</h3>
        <table>
            <tr><th>Video</th><th>mAP@0.3</th><th>Seg Ratio</th><th>Pred/GT</th></tr>
            {worst_rows}
        </table>
    </div>
    """


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------


def generate_benchmark_report(
    eval_results_path: str | Path,
    output_path: str | Path,
    predictions_path: str | Path | None = None,
    annotations_dir: str | Path | None = None,
) -> Path:
    """
    Generate the full HTML benchmark report.

    Args:
        eval_results_path: Path to eval_results.json
        output_path: Path to save the HTML report
        predictions_path: Optional path to mapped_predictions.jsonl (for timelines)
        annotations_dir: Optional path to annotations (for GT timelines)

    Returns:
        Path to the generated report
    """
    output_path = Path(output_path)

    # Load evaluation results
    with open(eval_results_path) as f:
        eval_results = json.load(f)

    # Load predictions for timelines
    predictions = None
    if predictions_path and Path(predictions_path).exists():
        predictions = {}
        with open(predictions_path) as f:
            for line in f:
                data = json.loads(line.strip())
                vid = data.get("video_id", data.get("_video_id", "unknown"))
                predictions.setdefault(vid, []).append(data)

    # Load GT for timelines
    gt_segments = None
    if annotations_dir and Path(annotations_dir).exists():
        try:
            from .load_epic_kitchens import EpicKitchensGT

            video_ids = list(eval_results.get("per_video", {}).keys())
            gt = EpicKitchensGT(annotations_dir, split="validation", video_filter=video_ids)
            gt_segments = {}
            for vid in video_ids:
                segs = gt.get_segments_for_video(vid)
                gt_segments[vid] = [
                    {
                        "start_t": s.start_t,
                        "end_t": s.end_t,
                        "verb": s.verb,
                        "noun": s.noun,
                        "narration": s.narration,
                    }
                    for s in segs
                ]
        except Exception as e:
            logger.warning(f"Could not load GT for timelines: {e}")

    # Load benchmark metadata
    meta_path = Path(eval_results_path).parent / "benchmark_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Generate sections
    summary = generate_summary_section(eval_results, meta)
    temporal = generate_temporal_section(eval_results)
    annotation = generate_annotation_section(eval_results)
    timelines = generate_timeline_section(eval_results, predictions, gt_segments)
    error_analysis = generate_error_analysis_section(eval_results)

    # Render
    html = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary_section=summary,
        temporal_section=temporal,
        annotation_section=annotation,
        timeline_section=timelines,
        error_analysis_section=error_analysis,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Benchmark report generated: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for benchmark report generation."""
    parser = argparse.ArgumentParser(
        description="Generate HTML benchmark report for EPIC-KITCHENS evaluation"
    )
    parser.add_argument(
        "--eval-results",
        type=str,
        required=True,
        help="Path to eval_results.json from evaluate.py",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to mapped_predictions.jsonl (for timeline visualization)",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/benchmark/epic_kitchens/annotations",
        help="Path to EPIC-KITCHENS annotations (for GT timelines)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: same dir as eval results)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_path = args.output
    if output_path is None:
        output_path = Path(args.eval_results).parent / "benchmark_report.html"

    report_path = generate_benchmark_report(
        eval_results_path=args.eval_results,
        output_path=output_path,
        predictions_path=args.predictions,
        annotations_dir=args.annotations_dir,
    )

    print(f"\nReport generated: file://{report_path.absolute()}")


if __name__ == "__main__":
    main()
