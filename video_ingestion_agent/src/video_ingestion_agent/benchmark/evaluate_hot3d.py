#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
HOT3D benchmark evaluator: temporal mAP@tIoU + object accuracy.

HOT3D ground truth carries (start_t, end_t, object_name) per per-object event;
no verb labels. We score:
    1. Temporal segmentation: mAP@tIoU{0.1, 0.3, 0.5, 0.7}, boundary precision/recall.
    2. Object identity: cosine similarity between agent's free-text ``object``
       and our ``object_name``, using sentence-transformers (the same family
       the EPIC adapter uses).

Usage::

    python -m video_ingestion_agent.benchmark.evaluate_hot3d \\
        --predictions runs/benchmark_hot3d/all_predictions.jsonl \\
        --ground-truth data/benchmark/hot3d/ground_truth_clips.jsonl \\
        --output runs/benchmark_hot3d/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video_ingestion_agent.benchmark.evaluate import (
    temporal_iou,
)
from video_ingestion_agent.benchmark.load_hot3d import Hot3dGT, HotGTSegment

logger = logging.getLogger(__name__)


@dataclass
class HotPredSegment:
    """Predicted segment as emitted by the agent (one row of clips_final.jsonl)."""

    video_id: str
    clip_id: str
    start_t: float
    end_t: float
    object: str
    action: str
    description: str
    score: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_predictions(path: Path) -> dict[str, list[HotPredSegment]]:
    """Read agent predictions JSONL, group by source-sequence-id (== video_id).

    The agent emits predictions keyed off ``video_path``. We derive
    ``video_id`` from the path stem when no explicit field is present.
    """
    preds: dict[str, list[HotPredSegment]] = {}
    with path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            vp = row.get("video_path", "")
            video_id = (
                row.get("video_id")
                or row.get("source_sequence_id")
                or Path(vp).parent.name
                or Path(vp).stem
            )
            seg = HotPredSegment(
                video_id=video_id,
                clip_id=row.get("clip_id", ""),
                start_t=float(row["start_t"]),
                end_t=float(row["end_t"]),
                object=row.get("object", ""),
                action=row.get("action", ""),
                description=row.get("description", ""),
                score=float(row.get("metadata", {}).get("confidence", 1.0)),
            )
            preds.setdefault(video_id, []).append(seg)
    for vid in preds:
        preds[vid].sort(key=lambda p: p.start_t)
    logger.info(
        "Loaded %d predicted segments across %d videos",
        sum(len(v) for v in preds.values()),
        len(preds),
    )
    return preds


# ---------------------------------------------------------------------------
# Temporal mAP — minimal standalone (mirrors the fallback in evaluate.py)
# ---------------------------------------------------------------------------


def _ap_at_tiou(
    preds: list[HotPredSegment],
    gts: list[HotGTSegment],
    tiou_threshold: float,
) -> float:
    """Average Precision at one tIoU threshold for a single video.

    Greedy-matched: predictions sorted by score descending, each prediction
    matches the highest-IoU unmatched GT above threshold.
    """
    if not preds:
        return 0.0
    if not gts:
        return 0.0

    iou = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou[i, j] = temporal_iou(p.start_t, p.end_t, g.start_t, g.end_t)

    sorted_idx = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    matched_gt: set[int] = set()
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for k, i in enumerate(sorted_idx):
        candidate_gts = [
            j for j in range(len(gts)) if j not in matched_gt and iou[i, j] >= tiou_threshold
        ]
        if not candidate_gts:
            fp[k] = 1
            continue
        best = max(candidate_gts, key=lambda j: iou[i, j])
        matched_gt.add(best)
        tp[k] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(len(gts), 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1)

    # 11-point interpolated AP (PASCAL-style) — robust across implementations
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        valid = recalls >= r
        ap += np.max(precisions[valid]) / 11 if np.any(valid) else 0.0
    return float(ap)


def compute_map_at_tiou(
    preds_by_video: dict[str, list[HotPredSegment]],
    gts_by_video: dict[str, list[HotGTSegment]],
    thresholds: list[float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for tiou in thresholds:
        per_video = []
        all_videos = set(preds_by_video) | set(gts_by_video)
        for vid in all_videos:
            per_video.append(
                _ap_at_tiou(
                    preds_by_video.get(vid, []),
                    gts_by_video.get(vid, []),
                    tiou,
                )
            )
        out[f"mAP@{tiou:.1f}"] = float(np.mean(per_video)) if per_video else 0.0
    out["mAP_avg"] = float(np.mean(list(out.values()))) if out else 0.0
    return out


# ---------------------------------------------------------------------------
# Boundary precision / recall — predictions that match a GT within ±tol_s
# ---------------------------------------------------------------------------


def boundary_metrics(
    preds_by_video: dict[str, list[HotPredSegment]],
    gts_by_video: dict[str, list[HotGTSegment]],
    tol_s: float = 1.0,
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    for vid in set(preds_by_video) | set(gts_by_video):
        preds = preds_by_video.get(vid, [])
        gts = gts_by_video.get(vid, [])
        gt_starts = [g.start_t for g in gts]
        gt_ends = [g.end_t for g in gts]
        matched_pred = [False] * len(preds)
        matched_gt_start = [False] * len(gt_starts)
        matched_gt_end = [False] * len(gt_ends)
        for i, p in enumerate(preds):
            ok = False
            for j, gt in enumerate(gt_starts):
                if not matched_gt_start[j] and abs(p.start_t - gt) <= tol_s:
                    matched_gt_start[j] = True
                    ok = True
                    break
            for j, gt in enumerate(gt_ends):
                if not matched_gt_end[j] and abs(p.end_t - gt) <= tol_s:
                    matched_gt_end[j] = True
                    ok = True
                    break
            if ok:
                matched_pred[i] = True
                tp += 1
            else:
                fp += 1
        fn += sum(1 for v in matched_gt_start if not v)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(f1),
    }


# ---------------------------------------------------------------------------
# Object accuracy via sentence-transformer cosine similarity
# ---------------------------------------------------------------------------


def object_accuracy(
    preds_by_video: dict[str, list[HotPredSegment]],
    gts_by_video: dict[str, list[HotGTSegment]],
    tiou_threshold: float = 0.3,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Mean cosine similarity between agent's `object` and GT's `object_name`,
    over pred-GT pairs that satisfy ``tIoU >= tiou_threshold`` (greedy-matched).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not available; skipping object-similarity metric.")
        return {"object_similarity_mean": 0.0, "object_matches": 0}

    model = SentenceTransformer(model_name)
    pairs: list[tuple[str, str]] = []

    for vid in set(preds_by_video) | set(gts_by_video):
        preds = preds_by_video.get(vid, [])
        gts = gts_by_video.get(vid, [])
        if not preds or not gts:
            continue
        iou = np.zeros((len(preds), len(gts)))
        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                iou[i, j] = temporal_iou(p.start_t, p.end_t, g.start_t, g.end_t)
        matched_gt: set[int] = set()
        for i in sorted(range(len(preds)), key=lambda x: -preds[x].score):
            cands = [
                j for j in range(len(gts)) if j not in matched_gt and iou[i, j] >= tiou_threshold
            ]
            if not cands:
                continue
            best = max(cands, key=lambda j: iou[i, j])
            matched_gt.add(best)
            pairs.append((preds[i].object, gts[best].object_name))

    if not pairs:
        return {"object_similarity_mean": 0.0, "object_matches": 0}

    pred_texts = [p or "" for p, _ in pairs]
    gt_texts = [g or "" for _, g in pairs]
    pred_emb = model.encode(pred_texts, normalize_embeddings=True, show_progress_bar=False)
    gt_emb = model.encode(gt_texts, normalize_embeddings=True, show_progress_bar=False)
    sims = (pred_emb * gt_emb).sum(axis=-1)

    return {
        "object_similarity_mean": float(np.mean(sims)),
        "object_similarity_median": float(np.median(sims)),
        "object_matches": int(len(pairs)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True, help="Agent JSONL.")
    ap.add_argument("--ground-truth", type=Path, required=True, help="HOT3D GT JSONL.")
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the eval_results JSON.",
    )
    ap.add_argument(
        "--tiou-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7],
    )
    ap.add_argument(
        "--object-match-tiou",
        type=float,
        default=0.3,
        help="tIoU floor for matching predictions to GT during object-similarity scoring.",
    )
    ap.add_argument(
        "--boundary-tol-s",
        type=float,
        default=1.0,
        help="±seconds tolerance when scoring boundary precision/recall.",
    )
    ap.add_argument("--video-filter", nargs="*", default=None)
    ap.add_argument(
        "--no-object-similarity",
        action="store_true",
        help="Skip the sentence-transformer object similarity metric.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gt = Hot3dGT(args.ground_truth, video_filter=args.video_filter)
    gts_by_video = {vid: gt.get_segments_for_video(vid) for vid in gt.get_all_video_ids()}
    preds_by_video = load_predictions(args.predictions)
    if args.video_filter:
        preds_by_video = {
            vid: segs for vid, segs in preds_by_video.items() if vid in set(args.video_filter)
        }

    print("\n=== Loaded ===", flush=True)
    print(f"  GT  : {sum(len(v) for v in gts_by_video.values())} segs / {len(gts_by_video)} videos")
    print(
        f"  Pred: {sum(len(v) for v in preds_by_video.values())} segs / {len(preds_by_video)} videos"
    )

    results: dict[str, dict] = {
        "metadata": {
            "ground_truth": str(args.ground_truth),
            "predictions": str(args.predictions),
            "n_gt_videos": len(gts_by_video),
            "n_pred_videos": len(preds_by_video),
            "n_gt_segments": sum(len(v) for v in gts_by_video.values()),
            "n_pred_segments": sum(len(v) for v in preds_by_video.values()),
            "tiou_thresholds": args.tiou_thresholds,
        },
        "temporal": compute_map_at_tiou(preds_by_video, gts_by_video, args.tiou_thresholds),
        "boundary": boundary_metrics(preds_by_video, gts_by_video, tol_s=args.boundary_tol_s),
    }
    if not args.no_object_similarity:
        results["object"] = object_accuracy(
            preds_by_video, gts_by_video, tiou_threshold=args.object_match_tiou
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote eval results → {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
