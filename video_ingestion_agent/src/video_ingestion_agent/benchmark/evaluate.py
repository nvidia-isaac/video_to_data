#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Evaluation script for EPIC-KITCHENS-100 benchmark.

Computes two tiers of metrics:
  1. Temporal segmentation: mAP@tIoU, boundary precision/recall, segmentation ratio
  2. Annotation accuracy: top-k verb/noun accuracy, semantic similarity, BERTScore

Usage:
    python -m video_ingestion_agent.benchmark.evaluate \\
        --predictions runs/benchmark_epic_kitchens/all_predictions.jsonl \\
        --annotations-dir data/benchmark/epic_kitchens/annotations \\
        --output runs/benchmark_epic_kitchens/eval_results.json

    # Skip slow metrics (BERTScore)
    python -m video_ingestion_agent.benchmark.evaluate \\
        --predictions runs/benchmark_epic_kitchens/all_predictions.jsonl \\
        --annotations-dir data/benchmark/epic_kitchens/annotations \\
        --skip-bertscore
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from video_ingestion_agent.benchmark.load_epic_kitchens import EpicKitchensGT, GTSegment

# Add C2-Action-Detection EvaluationCode to path for official mAP
_C2_EVAL_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "benchmark"
    / "epic_kitchens"
    / "C2-Action-Detection"
    / "EvaluationCode"
)
if _C2_EVAL_DIR.exists():
    sys.path.insert(0, str(_C2_EVAL_DIR))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PredSegment:
    """A predicted segment (from mapped predictions)."""

    video_id: str
    start_t: float
    end_t: float
    verb_class: int
    noun_class: int
    verb_label: str
    noun_label: str
    raw_action: str
    raw_object: str
    raw_description: str
    score: float = 1.0
    verb_similarity: float = 0.0
    noun_similarity: float = 0.0
    # Top-k alternatives: list of (class_id, label, similarity) tuples
    verb_top5: list[tuple[int, str, float]] = field(default_factory=list)
    noun_top5: list[tuple[int, str, float]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t

    @property
    def action_class(self) -> int:
        return self.verb_class * 300 + self.noun_class


@dataclass
class EvalResults:
    """Container for all evaluation results."""

    # Temporal metrics
    temporal: dict = field(default_factory=dict)

    # Annotation metrics
    annotation: dict = field(default_factory=dict)

    # Per-video details
    per_video: dict = field(default_factory=dict)

    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "temporal": self.temporal,
            "annotation": self.annotation,
            "per_video": self.per_video,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Temporal IoU computation
# ---------------------------------------------------------------------------


def temporal_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute temporal Intersection over Union between two segments."""
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0


def compute_iou_matrix(
    preds: list[PredSegment],
    gts: list[GTSegment],
) -> np.ndarray:
    """
    Compute pairwise temporal IoU matrix.

    Returns:
        np.ndarray of shape (len(preds), len(gts))
    """
    iou_mat = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_mat[i, j] = temporal_iou(p.start_t, p.end_t, g.start_t, g.end_t)
    return iou_mat


# ---------------------------------------------------------------------------
# Temporal Segmentation Metrics
# ---------------------------------------------------------------------------


def compute_map_at_tiou(
    all_preds: dict[str, list[PredSegment]],
    all_gts: dict[str, list[GTSegment]],
    tiou_thresholds: list[float] | None = None,
    annotations_pkl_path: str | Path | None = None,
) -> dict[str, float]:
    """
    Compute mean Average Precision at multiple tIoU thresholds.

    Tries to use the **official** C2-Action-Detection ANETdetection class
    (from data/benchmark/epic_kitchens/C2-Action-Detection/EvaluationCode/)
    for leaderboard-comparable results. Falls back to a standalone
    implementation if the official code is unavailable.

    Args:
        all_preds: Dict of video_id -> list of PredSegment
        all_gts: Dict of video_id -> list of GTSegment
        tiou_thresholds: IoU thresholds for matching
        annotations_pkl_path: Path to EPIC_100_validation.pkl (needed for official eval)

    Returns:
        Dict with mAP at each threshold and the average mAP
    """
    if tiou_thresholds is None:
        tiou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Try official C2 evaluation first
    official_results = _try_official_c2_eval(
        all_preds, all_gts, tiou_thresholds, annotations_pkl_path
    )
    if official_results is not None:
        return official_results

    # Fallback to standalone implementation
    logger.info("Using standalone mAP implementation (official C2 code not available)")
    return _compute_map_standalone(all_preds, all_gts, tiou_thresholds)


def _try_official_c2_eval(
    all_preds: dict[str, list[PredSegment]],
    all_gts: dict[str, list[GTSegment]],
    tiou_thresholds: list[float],
    annotations_pkl_path: str | Path | None,
) -> dict[str, float] | None:
    """
    Run the official C2-Action-Detection ANETdetection evaluation.

    Returns None if the official code is not available.
    """
    try:
        import pandas as pd
        from evaluate_detection_json_ek100 import ANETdetection
    except ImportError:
        logger.info("Official C2 EvaluationCode not importable, will use fallback")
        return None

    # Find annotations pkl
    if annotations_pkl_path is None:
        candidate = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "benchmark"
            / "epic_kitchens"
            / "annotations"
            / "EPIC_100_validation.pkl"
        )
        if candidate.exists():
            annotations_pkl_path = candidate
        else:
            logger.info("EPIC_100_validation.pkl not found, cannot use official eval")
            return None

    logger.info(f"Using official C2 ANETdetection with {annotations_pkl_path}")

    # Load GT annotations as DataFrame (official code expects this)
    annotations_df = pd.read_pickle(annotations_pkl_path)

    # Filter to only videos we have predictions for
    pred_video_ids = set(all_preds.keys())
    annotations_df = annotations_df[annotations_df["video_id"].isin(pred_video_ids)]

    if annotations_df.empty:
        logger.warning("No matching videos between predictions and annotations pkl")
        return None

    # Build submission dict in official C2 format
    submission_results: dict[str, list] = {}
    for video_id, preds in all_preds.items():
        entries = []
        for p in preds:
            entries.append(
                {
                    "segment": [p.start_t, p.end_t],
                    "verb": p.verb_class,
                    "noun": p.noun_class,
                    "action": f"{p.verb_class},{p.noun_class}",
                    "score": p.score,
                }
            )
        submission_results[video_id] = entries

    submission = {
        "version": "0.2",
        "challenge": "action_detection",
        "sls_pt": 0,
        "sls_tl": 0,
        "sls_td": 0,
        "results": submission_results,
    }

    # Run official evaluation for verb, noun, action
    thresholds = np.array(tiou_thresholds)
    results = {}

    for task in ["verb", "noun", "action"]:
        try:
            evaluator = ANETdetection(
                annotations_df,
                submission,
                tiou_thresholds=thresholds,
                label=task,
                num_nouns=300,
            )
            maps, avg_map = evaluator.evaluate()

            for i, t in enumerate(tiou_thresholds):
                results[f"{task}_mAP@{t}"] = float(maps[i])
            results[f"{task}_mAP_avg"] = float(avg_map)
        except Exception as e:
            logger.warning(f"Official C2 eval failed for {task}: {e}")
            for t in tiou_thresholds:
                results[f"{task}_mAP@{t}"] = 0.0
            results[f"{task}_mAP_avg"] = 0.0

    # Also provide the combined action mAP as the top-level mAP
    for t in tiou_thresholds:
        results[f"mAP@{t}"] = results.get(f"action_mAP@{t}", 0.0)
    results["mAP_avg"] = results.get("action_mAP_avg", 0.0)

    results["_eval_source"] = "official_C2_ANETdetection"

    return results


def _compute_map_standalone(
    all_preds: dict[str, list[PredSegment]],
    all_gts: dict[str, list[GTSegment]],
    tiou_thresholds: list[float],
) -> dict[str, float]:
    """Standalone mAP implementation (fallback when official C2 code is unavailable)."""
    results = {}

    for threshold in tiou_thresholds:
        all_matches = []

        for video_id in all_gts:
            preds = all_preds.get(video_id, [])
            gts = all_gts[video_id]

            if not preds or not gts:
                continue

            preds_sorted = sorted(preds, key=lambda p: p.score, reverse=True)
            iou_mat = compute_iou_matrix(preds_sorted, gts)

            gt_matched = set()
            for i, pred in enumerate(preds_sorted):
                best_iou = 0.0
                best_gt_idx = -1

                for j in range(len(gts)):
                    if j in gt_matched:
                        continue
                    if iou_mat[i, j] > best_iou:
                        best_iou = iou_mat[i, j]
                        best_gt_idx = j

                if best_iou >= threshold and best_gt_idx >= 0:
                    all_matches.append((pred.score, True))
                    gt_matched.add(best_gt_idx)
                else:
                    all_matches.append((pred.score, False))

        if not all_matches:
            results[f"mAP@{threshold}"] = 0.0
            continue

        all_matches.sort(key=lambda x: x[0], reverse=True)

        total_gt = sum(len(gts) for gts in all_gts.values())
        tp_cumsum = 0
        precisions = []
        recalls = []

        for _score, is_match in all_matches:
            if is_match:
                tp_cumsum += 1
            precision = tp_cumsum / (len(precisions) + 1)
            recall = tp_cumsum / total_gt if total_gt > 0 else 0
            precisions.append(precision)
            recalls.append(recall)

        ap = _interpolated_ap(precisions, recalls)
        results[f"mAP@{threshold}"] = ap

    map_values = [results[f"mAP@{t}"] for t in tiou_thresholds]
    results["mAP_avg"] = float(np.mean(map_values)) if map_values else 0.0
    results["_eval_source"] = "standalone_fallback"

    return results


def _interpolated_ap(precisions: list[float], recalls: list[float]) -> float:
    """Compute interpolated Average Precision (VOC-style)."""
    if not precisions:
        return 0.0

    mprec = np.hstack([[0], precisions, [0]])
    mrec = np.hstack([[0], recalls, [1]])

    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])

    return float(ap)


def compute_boundary_metrics(
    all_preds: dict[str, list[PredSegment]],
    all_gts: dict[str, list[GTSegment]],
    tolerances: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute boundary precision and recall.

    For each GT boundary (start/end), check if a predicted boundary
    falls within the tolerance window.

    Args:
        all_preds: Dict of video_id -> list of PredSegment
        all_gts: Dict of video_id -> list of GTSegment
        tolerances: Tolerance windows in seconds

    Returns:
        Dict with boundary metrics for each tolerance
    """
    if tolerances is None:
        tolerances = [1.0, 2.0]

    results = {}

    for tolerance in tolerances:
        total_gt_boundaries = 0
        total_pred_boundaries = 0
        gt_matched = 0
        pred_matched = 0

        for video_id in all_gts:
            preds = all_preds.get(video_id, [])
            gts = all_gts[video_id]

            # Collect boundaries
            gt_boundaries = []
            for g in gts:
                gt_boundaries.extend([g.start_t, g.end_t])
            gt_boundaries = sorted(set(gt_boundaries))

            pred_boundaries = []
            for p in preds:
                pred_boundaries.extend([p.start_t, p.end_t])
            pred_boundaries = sorted(set(pred_boundaries))

            total_gt_boundaries += len(gt_boundaries)
            total_pred_boundaries += len(pred_boundaries)

            # Check recall: for each GT boundary, is there a pred boundary nearby?
            pred_used_for_recall = set()
            for gt_b in gt_boundaries:
                for k, pred_b in enumerate(pred_boundaries):
                    if k in pred_used_for_recall:
                        continue
                    if abs(gt_b - pred_b) <= tolerance:
                        gt_matched += 1
                        pred_used_for_recall.add(k)
                        break

            # Check precision: for each pred boundary, is there a GT boundary nearby?
            gt_used_for_precision = set()
            for pred_b in pred_boundaries:
                for k, gt_b in enumerate(gt_boundaries):
                    if k in gt_used_for_precision:
                        continue
                    if abs(pred_b - gt_b) <= tolerance:
                        pred_matched += 1
                        gt_used_for_precision.add(k)
                        break

        precision = pred_matched / total_pred_boundaries if total_pred_boundaries > 0 else 0.0
        recall = gt_matched / total_gt_boundaries if total_gt_boundaries > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[f"boundary@{tolerance}s"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gt_boundaries": total_gt_boundaries,
            "pred_boundaries": total_pred_boundaries,
        }

    return results


def compute_segmentation_ratio(
    all_preds: dict[str, list[PredSegment]],
    all_gts: dict[str, list[GTSegment]],
) -> dict[str, float]:
    """
    Compute segmentation ratio: len(predicted) / len(GT) per video.

    A ratio > 1 means over-segmentation, < 1 means under-segmentation.
    """
    ratios = []
    per_video = {}

    for video_id in all_gts:
        n_pred = len(all_preds.get(video_id, []))
        n_gt = len(all_gts[video_id])
        ratio = n_pred / n_gt if n_gt > 0 else 0.0
        ratios.append(ratio)
        per_video[video_id] = {
            "n_pred": n_pred,
            "n_gt": n_gt,
            "ratio": ratio,
        }

    return {
        "mean_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "median_ratio": float(np.median(ratios)) if ratios else 0.0,
        "std_ratio": float(np.std(ratios)) if ratios else 0.0,
        "over_segmented": sum(1 for r in ratios if r > 1.5),
        "under_segmented": sum(1 for r in ratios if r < 0.5),
        "well_segmented": sum(1 for r in ratios if 0.5 <= r <= 1.5),
        "per_video": per_video,
    }


# ---------------------------------------------------------------------------
# Annotation Accuracy Metrics
# ---------------------------------------------------------------------------


def compute_matched_accuracy(
    all_preds: dict[str, list[PredSegment]],
    all_gts: dict[str, list[GTSegment]],
    tiou_threshold: float = 0.3,
) -> dict[str, float]:
    """
    Compute top-1 and top-5 verb/noun accuracy for matched segments.

    First matches predicted segments to GT using greedy IoU matching,
    then checks if the mapped verb/noun class matches.

    Args:
        all_preds: Dict of video_id -> list of PredSegment
        all_gts: Dict of video_id -> list of GTSegment
        tiou_threshold: Minimum IoU for a valid match

    Returns:
        Dict with accuracy metrics
    """
    verb_correct_1 = 0
    noun_correct_1 = 0
    verb_correct_5 = 0
    noun_correct_5 = 0
    total_matched = 0

    for video_id in all_gts:
        preds = all_preds.get(video_id, [])
        gts = all_gts[video_id]

        if not preds or not gts:
            continue

        # Compute IoU matrix
        iou_mat = compute_iou_matrix(preds, gts)

        # Greedy matching (best IoU first)
        gt_matched = set()
        pred_matched = set()

        # Find all pairs above threshold, sort by IoU
        pairs = []
        for i in range(len(preds)):
            for j in range(len(gts)):
                if iou_mat[i, j] >= tiou_threshold:
                    pairs.append((iou_mat[i, j], i, j))
        pairs.sort(reverse=True)

        for _iou, pi, gi in pairs:
            if pi in pred_matched or gi in gt_matched:
                continue
            pred_matched.add(pi)
            gt_matched.add(gi)

            pred = preds[pi]
            gt = gts[gi]
            total_matched += 1

            # Top-1 accuracy
            if pred.verb_class == gt.verb_class:
                verb_correct_1 += 1
            if pred.noun_class == gt.noun_class:
                noun_correct_1 += 1

            # Top-5 accuracy: check if GT class appears among the top-5
            # predicted classes.  Falls back to top-1 if top-5 data is
            # not available (empty list).
            verb_top5_ids = (
                {c for c, _, _ in pred.verb_top5} if pred.verb_top5 else {pred.verb_class}
            )
            noun_top5_ids = (
                {c for c, _, _ in pred.noun_top5} if pred.noun_top5 else {pred.noun_class}
            )

            if gt.verb_class in verb_top5_ids:
                verb_correct_5 += 1
            if gt.noun_class in noun_top5_ids:
                noun_correct_5 += 1

    return {
        "total_matched": total_matched,
        "verb_top1_acc": verb_correct_1 / total_matched if total_matched > 0 else 0.0,
        "noun_top1_acc": noun_correct_1 / total_matched if total_matched > 0 else 0.0,
        "verb_top5_acc": verb_correct_5 / total_matched if total_matched > 0 else 0.0,
        "noun_top5_acc": noun_correct_5 / total_matched if total_matched > 0 else 0.0,
    }


def compute_soft_match_rate(
    all_preds: dict[str, list[PredSegment]],
    all_gts: dict[str, list[GTSegment]],
    tiou_threshold: float = 0.3,
) -> dict[str, float]:
    """
    Compute soft match rate: does GT verb/noun appear as substring in predicted text?

    This is a fairer metric for zero-shot free-text pipeline output.
    """
    verb_soft_match = 0
    noun_soft_match = 0
    total_matched = 0

    for video_id in all_gts:
        preds = all_preds.get(video_id, [])
        gts = all_gts[video_id]

        if not preds or not gts:
            continue

        iou_mat = compute_iou_matrix(preds, gts)

        # Greedy matching
        gt_matched_set = set()
        pred_matched_set = set()
        pairs = []
        for i in range(len(preds)):
            for j in range(len(gts)):
                if iou_mat[i, j] >= tiou_threshold:
                    pairs.append((iou_mat[i, j], i, j))
        pairs.sort(reverse=True)

        for _iou, pi, gi in pairs:
            if pi in pred_matched_set or gi in gt_matched_set:
                continue
            pred_matched_set.add(pi)
            gt_matched_set.add(gi)

            pred = preds[pi]
            gt = gts[gi]
            total_matched += 1

            # Check if GT verb appears in predicted action/description
            pred_text = f"{pred.raw_action} {pred.raw_description}".lower()
            if gt.verb.lower() in pred_text:
                verb_soft_match += 1

            # Check if GT noun appears in predicted object/description
            pred_obj_text = f"{pred.raw_object} {pred.raw_description}".lower()
            if gt.noun.lower() in pred_obj_text:
                noun_soft_match += 1

    return {
        "total_matched": total_matched,
        "verb_soft_match_rate": verb_soft_match / total_matched if total_matched > 0 else 0.0,
        "noun_soft_match_rate": noun_soft_match / total_matched if total_matched > 0 else 0.0,
    }


def compute_semantic_similarity(
    all_preds: dict[str, list[PredSegment]],
    all_gts: dict[str, list[GTSegment]],
    tiou_threshold: float = 0.3,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, float]:
    """
    Compute semantic similarity between predicted descriptions and GT narrations.

    Uses sentence-transformers cosine similarity.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping semantic similarity")
        return {"error": "sentence-transformers not installed"}

    model = SentenceTransformer(model_name)

    pred_texts = []
    gt_texts = []

    for video_id in all_gts:
        preds = all_preds.get(video_id, [])
        gts = all_gts[video_id]

        if not preds or not gts:
            continue

        iou_mat = compute_iou_matrix(preds, gts)

        gt_matched_set = set()
        pred_matched_set = set()
        pairs = []
        for i in range(len(preds)):
            for j in range(len(gts)):
                if iou_mat[i, j] >= tiou_threshold:
                    pairs.append((iou_mat[i, j], i, j))
        pairs.sort(reverse=True)

        for _iou, pi, gi in pairs:
            if pi in pred_matched_set or gi in gt_matched_set:
                continue
            pred_matched_set.add(pi)
            gt_matched_set.add(gi)

            pred = preds[pi]
            gt = gts[gi]

            pred_text = (
                pred.raw_description
                if pred.raw_description
                else f"{pred.raw_action} {pred.raw_object}"
            )
            pred_texts.append(pred_text)
            gt_texts.append(gt.narration)

    if not pred_texts:
        return {"mean_similarity": 0.0, "n_pairs": 0}

    # Compute embeddings
    pred_embs = model.encode(pred_texts, normalize_embeddings=True, show_progress_bar=False)
    gt_embs = model.encode(gt_texts, normalize_embeddings=True, show_progress_bar=False)

    # Pairwise cosine similarity
    similarities = np.sum(pred_embs * gt_embs, axis=1)

    return {
        "mean_similarity": float(np.mean(similarities)),
        "median_similarity": float(np.median(similarities)),
        "std_similarity": float(np.std(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "n_pairs": len(similarities),
    }


def compute_bertscore(
    all_preds: dict[str, list[PredSegment]],
    all_gts: dict[str, list[GTSegment]],
    tiou_threshold: float = 0.3,
) -> dict[str, float]:
    """
    Compute BERTScore between predicted descriptions and GT narrations.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.warning("bert-score not installed, skipping BERTScore")
        return {"error": "bert-score not installed"}

    pred_texts = []
    gt_texts = []

    for video_id in all_gts:
        preds = all_preds.get(video_id, [])
        gts = all_gts[video_id]

        if not preds or not gts:
            continue

        iou_mat = compute_iou_matrix(preds, gts)

        gt_matched_set = set()
        pred_matched_set = set()
        pairs = []
        for i in range(len(preds)):
            for j in range(len(gts)):
                if iou_mat[i, j] >= tiou_threshold:
                    pairs.append((iou_mat[i, j], i, j))
        pairs.sort(reverse=True)

        for _iou, pi, gi in pairs:
            if pi in pred_matched_set or gi in gt_matched_set:
                continue
            pred_matched_set.add(pi)
            gt_matched_set.add(gi)

            pred = preds[pi]
            gt = gts[gi]

            pred_text = (
                pred.raw_description
                if pred.raw_description
                else f"{pred.raw_action} {pred.raw_object}"
            )
            pred_texts.append(pred_text)
            gt_texts.append(gt.narration)

    if not pred_texts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pairs": 0}

    P, R, F1 = bert_score_fn(pred_texts, gt_texts, lang="en", verbose=False)

    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
        "n_pairs": len(pred_texts),
    }


# ---------------------------------------------------------------------------
# Loading predictions
# ---------------------------------------------------------------------------


def load_mapped_predictions(predictions_file: str | Path) -> dict[str, list[PredSegment]]:
    """
    Load mapped predictions from JSONL file into per-video dict.

    Supports both raw pipeline output and adapter-mapped output.
    """
    predictions_file = Path(predictions_file)
    by_video: dict[str, list[PredSegment]] = defaultdict(list)

    with open(predictions_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Determine video_id
            video_id = data.get("video_id", data.get("_video_id", ""))
            if not video_id:
                video_path = data.get("video_path", "")
                video_id = Path(video_path).stem if video_path else "unknown"

            # Parse top-5 lists -- each entry is [class_id, label, sim]
            verb_top5_raw = data.get("verb_top5", [])
            noun_top5_raw = data.get("noun_top5", [])
            verb_top5 = [
                (int(e[0]), str(e[1]), float(e[2]))
                for e in verb_top5_raw
                if isinstance(e, (list, tuple)) and len(e) >= 3
            ]
            noun_top5 = [
                (int(e[0]), str(e[1]), float(e[2]))
                for e in noun_top5_raw
                if isinstance(e, (list, tuple)) and len(e) >= 3
            ]

            pred = PredSegment(
                video_id=video_id,
                start_t=data.get("start_t", 0.0),
                end_t=data.get("end_t", 0.0),
                verb_class=data.get("verb_class", -1),
                noun_class=data.get("noun_class", -1),
                verb_label=data.get("verb_label", ""),
                noun_label=data.get("noun_label", ""),
                raw_action=data.get("raw_action", data.get("action", "")),
                raw_object=data.get("raw_object", data.get("object", "")),
                raw_description=data.get("raw_description", data.get("description", "")),
                score=data.get("score", 1.0),
                verb_similarity=data.get("verb_similarity", 0.0),
                noun_similarity=data.get("noun_similarity", 0.0),
                verb_top5=verb_top5,
                noun_top5=noun_top5,
            )
            by_video[video_id].append(pred)

    # Sort by start time
    for vid in by_video:
        by_video[vid].sort(key=lambda p: p.start_t)

    logger.info(
        f"Loaded {sum(len(v) for v in by_video.values())} predictions across {len(by_video)} videos"
    )
    return dict(by_video)


# ---------------------------------------------------------------------------
# Main evaluation orchestrator
# ---------------------------------------------------------------------------


def run_evaluation(
    predictions_file: str | Path,
    annotations_dir: str | Path,
    tiou_thresholds: list[float] | None = None,
    boundary_tolerances: list[float] | None = None,
    skip_bertscore: bool = False,
    skip_semantic: bool = False,
) -> EvalResults:
    """
    Run the full evaluation pipeline.

    Args:
        predictions_file: Path to mapped predictions JSONL
        annotations_dir: Path to EPIC-KITCHENS annotations
        tiou_thresholds: tIoU thresholds for mAP
        boundary_tolerances: Boundary matching tolerances in seconds
        skip_bertscore: Skip BERTScore computation (slow)
        skip_semantic: Skip semantic similarity (needs sentence-transformers)

    Returns:
        EvalResults with all metrics
    """
    if tiou_thresholds is None:
        tiou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    if boundary_tolerances is None:
        boundary_tolerances = [1.0, 2.0]

    results = EvalResults()

    # Load data
    logger.info("Loading predictions...")
    all_preds = load_mapped_predictions(predictions_file)

    logger.info("Loading ground truth...")
    video_ids = list(all_preds.keys())
    gt = EpicKitchensGT(annotations_dir, split="validation", video_filter=video_ids)
    all_gts = {vid: gt.get_segments_for_video(vid) for vid in video_ids}

    # Filter out videos with no GT
    all_gts = {vid: segs for vid, segs in all_gts.items() if segs}

    results.metadata = {
        "n_pred_videos": len(all_preds),
        "n_gt_videos": len(all_gts),
        "n_pred_segments": sum(len(v) for v in all_preds.values()),
        "n_gt_segments": sum(len(v) for v in all_gts.values()),
        "tiou_thresholds": tiou_thresholds,
        "boundary_tolerances": boundary_tolerances,
    }

    # --- Temporal Metrics ---
    logger.info("Computing mAP@tIoU...")
    annotations_pkl = Path(annotations_dir) / "EPIC_100_validation.pkl"
    map_results = compute_map_at_tiou(
        all_preds,
        all_gts,
        tiou_thresholds,
        annotations_pkl_path=annotations_pkl if annotations_pkl.exists() else None,
    )
    results.temporal["mAP"] = map_results

    logger.info("Computing boundary metrics...")
    boundary_results = compute_boundary_metrics(all_preds, all_gts, boundary_tolerances)
    results.temporal["boundary"] = boundary_results

    logger.info("Computing segmentation ratio...")
    seg_ratio = compute_segmentation_ratio(all_preds, all_gts)
    results.temporal["segmentation_ratio"] = {
        k: v for k, v in seg_ratio.items() if k != "per_video"
    }

    # --- Annotation Metrics ---
    logger.info("Computing matched accuracy...")
    accuracy = compute_matched_accuracy(all_preds, all_gts)
    results.annotation["accuracy"] = accuracy

    logger.info("Computing soft match rate...")
    soft_match = compute_soft_match_rate(all_preds, all_gts)
    results.annotation["soft_match"] = soft_match

    if not skip_semantic:
        logger.info("Computing semantic similarity...")
        sem_sim = compute_semantic_similarity(all_preds, all_gts)
        results.annotation["semantic_similarity"] = sem_sim

    if not skip_bertscore:
        logger.info("Computing BERTScore...")
        bert = compute_bertscore(all_preds, all_gts)
        results.annotation["bertscore"] = bert

    # --- Per-video details ---
    for video_id in sorted(all_gts.keys()):
        preds_v = {video_id: all_preds.get(video_id, [])}
        gts_v = {video_id: all_gts[video_id]}

        video_map = compute_map_at_tiou(preds_v, gts_v, [0.3, 0.5])
        video_seg_ratio = compute_segmentation_ratio(preds_v, gts_v)

        results.per_video[video_id] = {
            "n_pred": len(all_preds.get(video_id, [])),
            "n_gt": len(all_gts[video_id]),
            "seg_ratio": video_seg_ratio.get("mean_ratio", 0.0),
            "mAP@0.3": video_map.get("mAP@0.3", 0.0),
            "mAP@0.5": video_map.get("mAP@0.5", 0.0),
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for benchmark evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate video_ingestion_agent predictions against EPIC-KITCHENS-100 GT"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to mapped predictions JSONL (from adapter or all_predictions.jsonl)",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/benchmark/epic_kitchens/annotations",
        help="Path to EPIC-KITCHENS annotations directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore computation (slow)",
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip semantic similarity computation",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Run evaluation
    results = run_evaluation(
        predictions_file=args.predictions,
        annotations_dir=args.annotations_dir,
        skip_bertscore=args.skip_bertscore,
        skip_semantic=args.skip_semantic,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EPIC-KITCHENS-100 Benchmark Results")
    print("=" * 60)

    print("\n--- Temporal Segmentation ---")
    if "mAP" in results.temporal:
        for key, val in results.temporal["mAP"].items():
            print(f"  {key}: {val:.4f}")

    if "boundary" in results.temporal:
        for key, val in results.temporal["boundary"].items():
            print(f"  {key}:")
            for k2, v2 in val.items():
                print(f"    {k2}: {v2:.4f}" if isinstance(v2, float) else f"    {k2}: {v2}")

    if "segmentation_ratio" in results.temporal:
        for key, val in results.temporal["segmentation_ratio"].items():
            print(
                f"  seg_ratio.{key}: {val:.4f}"
                if isinstance(val, float)
                else f"  seg_ratio.{key}: {val}"
            )

    print("\n--- Annotation Accuracy ---")
    if "accuracy" in results.annotation:
        for key, val in results.annotation["accuracy"].items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    if "soft_match" in results.annotation:
        for key, val in results.annotation["soft_match"].items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    if "semantic_similarity" in results.annotation:
        for key, val in results.annotation["semantic_similarity"].items():
            print(
                f"  semantic.{key}: {val:.4f}"
                if isinstance(val, float)
                else f"  semantic.{key}: {val}"
            )

    if "bertscore" in results.annotation:
        for key, val in results.annotation["bertscore"].items():
            print(
                f"  bertscore.{key}: {val:.4f}"
                if isinstance(val, float)
                else f"  bertscore.{key}: {val}"
            )

    # Save results
    output_path = args.output
    if output_path is None:
        pred_dir = Path(args.predictions).parent
        output_path = pred_dir / "eval_results.json"

    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
