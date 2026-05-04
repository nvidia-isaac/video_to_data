#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Prediction-to-submission adapter for EPIC-KITCHENS-100 benchmark.

Maps free-text pipeline predictions (ClipContext.action, .object, .description)
to EPIC-KITCHENS verb_class/noun_class IDs using sentence-transformer embeddings
with **multi-vector synonym matching**.

For each EPIC class, every known synonym/instance is embedded individually.
A query is matched to the class whose *best* synonym has the highest cosine
similarity (max-sim retrieval).

Also converts predictions to C2-Action-Detection submission format.

Usage:
    from video_ingestion_agent.benchmark.adapter import EpicKitchensAdapter
    adapter = EpicKitchensAdapter(
        verb_classes=gt.verb_classes,
        noun_classes=gt.noun_classes,
        verb_instances=gt.verb_instances,
        noun_instances=gt.noun_instances,
    )
    submission = adapter.convert_predictions(predictions)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------

# Common colour / material / size adjectives that the VLM prepends to objects
# but that are irrelevant for EPIC noun matching.
_STRIP_ADJECTIVES = {
    "metal",
    "metallic",
    "black",
    "white",
    "blue",
    "red",
    "green",
    "yellow",
    "brown",
    "gray",
    "grey",
    "silver",
    "gold",
    "orange",
    "pink",
    "purple",
    "clear",
    "transparent",
    "dark",
    "light",
    "bright",
    "wooden",
    "plastic",
    "glass",
    "rubber",
    "stainless",
    "steel",
    "ceramic",
    "aluminum",
    "aluminium",
    "small",
    "large",
    "big",
    "tiny",
    "tall",
    "short",
    "thin",
    "thick",
    "round",
    "flat",
    "square",
    "rectangular",
    "remaining",
    "patterned",
    "striped",
    "dirty",
    "clean",
}

# Prepositions used to split compound object descriptions
_SPLIT_PREPOSITIONS = {
    "on",
    "in",
    "with",
    "from",
    "of",
    "inside",
    "under",
    "over",
    "near",
    "next",
    "behind",
    "between",
    "wrapped",
}


def normalize_verb_text(text: str) -> str:
    """
    Normalize a free-text verb prediction to better match EPIC instance formats.

    - Lower-cases
    - Converts multi-word phrasal verbs to hyphenated form
      ("turn on" -> "turn-on", "pick up" -> "pick-up")
    """
    text = text.strip().lower()
    # Common two-word phrasal verbs -> hyphenated
    text = re.sub(r"\b(\w+)\s+(on|off|up|down|in|out|away|over|back|into|for)\b", r"\1-\2", text)
    return text


# ---------------------------------------------------------------------------
# Verb alias table  –  maps VLM-produced verbs to EPIC-compatible verbs
# ---------------------------------------------------------------------------

# Static aliases: VLM verb -> EPIC verb (applied *before* embedding lookup).
# These handle the common case where the VLM uses a more specific verb than
# EPIC's coarse taxonomy expects.
#
# A value of ``None`` means "needs context-aware disambiguation" and is
# handled by ``_contextual_verb_remap`` instead.
_VERB_ALIASES: dict[str, str | None] = {
    # --- VLM is too granular; EPIC annotators use "take" for all of these ---
    "lift": "take",
    "carry": "take",
    "hold": "take",
    "pick": "take",
    "picking-up": "pick-up",  # tense normalisation
    # --- "place" needs context: put-down vs insert ---
    "place": None,  # -> _contextual_verb_remap
    # --- "turn" needs context: turn-on vs turn-off vs rotate ---
    "turn": None,  # -> _contextual_verb_remap
    # --- "push" is usually "move" in EPIC kitchen context ---
    "push": "move",
    # --- "adjust" is usually "move" ---
    "adjust": "move",
    # --- "scrub" is a wash variant ---
    "scrub": "wash",
    # --- VLM gerund forms -> base form ---
    "wiping": "wipe",
    "rinsing": "rinse",
    "washing": "wash",
    "cutting": "cut",
    "grabbing": "grab",
    "lifting": "take",
    "holding": "take",
    "carrying": "take",
    "opening": "open",
    "stirring": "stir",
    "placing": None,  # -> _contextual_verb_remap (same as "place")
    "chopping": "chop",
    "retrieving": "take",
    "pouring": "pour",
    "removing": "remove",
    "peeling": "peel",
    "repositioning": "move",
    "viewing": "look",
    "unwrapping": "unwrap",
    "scrubbing": "wash",
    "adjusting": "move",
    "turning": None,  # -> _contextual_verb_remap (same as "turn")
    "closing": "close",
    "shaking": "shake",
    "drying": "dry",
    "squeezing": "squeeze",
    "scooping": "scoop",
    "mixing": "mix",
    "flipping": "flip",
    "sprinkling": "sprinkle",
    "spreading": "spread",
    "folding": "fold",
    "sorting": "sort",
    "filling": "fill",
    "draining": "drain",
    "seasoning": "season",
    "pressing": "press",
    "scraping": "scrape",
    # --- non-manipulation verbs -> closest EPIC class ---
    "approach": "walk",
    "walk-toward": "walk",
    "walk-towards": "walk",
    "inspect": "check",
    "look-at": "look",
    "observe": "look",
    "manipulate": "move",
    # --- compound VLM actions -> pick the primary verb ---
    "pick-up-and-place": "take",
    "grab-and-place": "take",
    "empty": "empty",
    # --- compound phrase aliases from 47-video benchmark ---
    # Short phrasal verbs (normalization hyphenates the particle)
    "reaching-for": "take",
    "pulling-out": "take-out",
    "placing-down": "put-down",
}

# Prepositions that signal "put INTO" (class 5 = insert) rather than
# "put DOWN" (class 1 = put) when the raw action is "place".
_INSERT_PREPOSITIONS = re.compile(r"\b(?:into|in(?:to)?|inside|within)\b", re.IGNORECASE)

# Cues that signal "turn ON" vs "turn OFF" in descriptions.
_TURN_ON_CUES = re.compile(
    r"\b(?:turn(?:s|ed|ing)?\s+on|start|activate|ignite|begin|switch(?:es|ed|ing)?\s+on)\b",
    re.IGNORECASE,
)
_TURN_OFF_CUES = re.compile(
    r"\b(?:turn(?:s|ed|ing)?\s+off|stop|shut|deactivate|switch(?:es|ed|ing)?\s+off)\b",
    re.IGNORECASE,
)


def _contextual_verb_remap(
    raw_action: str,
    raw_object: str,
    raw_description: str,
) -> str:
    """
    Context-aware verb remapping for ambiguous VLM verbs.

    Uses the object and description text to disambiguate:
    - "place" -> "put-down" (default) vs "put-into" / "put-in" (if
      description contains "into"/"inside")
    - "turn" -> "turn-on" / "turn-off" (if description gives direction)
      or "turn" (rotate) as fallback

    Returns:
        The remapped verb string to use for embedding lookup.
    """
    action = raw_action.strip().lower()
    context = f"{raw_object} {raw_description}".lower()

    if action in ("place", "placing"):
        # Check if the description implies insertion ("into", "in", "inside")
        if _INSERT_PREPOSITIONS.search(context):
            return "put-into"
        return "put-down"

    if action in ("turn", "turning"):
        # Check description for on/off cues
        if _TURN_ON_CUES.search(context):
            return "turn-on"
        if _TURN_OFF_CUES.search(context):
            return "turn-off"
        # No on/off cue — check for water / faucet / tap context
        # (in kitchen videos, turning a faucet is almost always turn-on/off)
        if re.search(r"\b(?:faucet|tap|water|knob)\b", context):
            # Default to turn-on if ambiguous; EPIC annotators lean this way
            return "turn-on"
        # True rotation (e.g., turning a lid)
        return "turn"

    # Fallback – should not normally be reached if _VERB_ALIASES is correct
    return action


def resolve_verb_text(
    raw_action: str,
    raw_object: str = "",
    raw_description: str = "",
) -> str:
    """
    Resolve a VLM-produced action string into an EPIC-compatible verb.

    Pipeline:
    1. Normalise (lower-case, hyphenate phrasal verbs).
    2. If the normalised verb is in ``_VERB_ALIASES``:
       a. Static alias (str) -> use directly.
       b. ``None`` -> call ``_contextual_verb_remap`` for disambiguation.
    3. If compound action (contains comma / "and"), extract the first verb.
    4. Otherwise return the normalised text unchanged (embedding handles it).

    Args:
        raw_action: The ``action`` field from VLM output.
        raw_object: The ``object`` field (used for context).
        raw_description: The ``description`` field (used for context).

    Returns:
        Resolved verb string ready for embedding lookup.
    """
    text = normalize_verb_text(raw_action)
    if not text:
        return text

    # Handle compound actions: "grab, unfold, and place" -> "grab"
    if "," in text or " and " in text:
        first_verb = re.split(r"[,]|\band\b", text)[0].strip()
        if first_verb:
            text = first_verb
            # Re-normalise the extracted first verb
            text = normalize_verb_text(text)

    # Apply alias table
    if text in _VERB_ALIASES:
        alias = _VERB_ALIASES[text]
        if alias is None:
            resolved = _contextual_verb_remap(raw_action, raw_object, raw_description)
            logger.debug(
                "Contextual verb remap: %r -> %r (obj=%r)",
                raw_action,
                resolved,
                raw_object,
            )
            return resolved
        logger.debug("Verb alias: %r -> %r", text, alias)
        return alias

    return text


def extract_noun_candidates(text: str) -> list[str]:
    """
    Extract candidate noun phrases from a descriptive object string.

    Pipeline objects look like:
        "metal baking tray"
        "pizza slice on patterned plate"
        "brown paper bag"

    Strategy:
    1. Split on prepositions to get sub-phrases.
    2. For each sub-phrase, strip leading colour/material adjectives.
    3. Return all non-empty candidates (the *original* text is always first).
    """
    text = text.strip().lower()
    if not text:
        return []

    candidates: list[str] = [text]  # always try the full text first

    # Split by prepositions
    parts = re.split(r"\b(?:" + "|".join(_SPLIT_PREPOSITIONS) + r")\b", text)
    for part in parts:
        part = part.strip()
        if not part or part == text:
            continue
        candidates.append(part)

    # For each candidate, also produce a version with adjectives stripped
    expanded: list[str] = []
    for cand in candidates:
        expanded.append(cand)
        words = cand.split()
        # Strip leading adjectives
        while len(words) > 1 and words[0] in _STRIP_ADJECTIVES:
            words = words[1:]
        stripped = " ".join(words)
        if stripped != cand and stripped:
            expanded.append(stripped)

    # De-duplicate while preserving order
    seen = set()
    unique: list[str] = []
    for c in expanded:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MappedPrediction:
    """A pipeline prediction mapped to EPIC-KITCHENS class IDs."""

    # Original prediction
    video_id: str
    start_t: float
    end_t: float
    raw_action: str
    raw_object: str
    raw_description: str

    # Mapped classes
    verb_class: int
    noun_class: int
    verb_label: str
    noun_label: str
    verb_similarity: float  # cosine similarity of the match
    noun_similarity: float

    # Top-k alternatives
    verb_top5: list[tuple[int, str, float]] = field(default_factory=list)  # (class, label, sim)
    noun_top5: list[tuple[int, str, float]] = field(default_factory=list)

    # Combined action class (verb * num_nouns + noun)
    @property
    def action_class(self) -> int:
        return self.verb_class * 300 + self.noun_class

    # Confidence score (average of verb and noun similarity)
    @property
    def score(self) -> float:
        return (self.verb_similarity + self.noun_similarity) / 2.0


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class EpicKitchensAdapter:
    """
    Adapts free-text pipeline predictions to EPIC-KITCHENS class IDs.

    Uses **multi-vector synonym matching**: for each EPIC class, every known
    synonym/instance from the class vocabulary CSV is embedded individually.
    A query is matched to the class whose *best* synonym has the highest
    cosine similarity (max-sim).

    Also applies text preprocessing:
    - Verbs: normalise phrasal verbs ("turn on" -> "turn-on")
    - Nouns: split compound descriptions, strip colour/material adjectives,
      and try each candidate sub-phrase.
    """

    def __init__(
        self,
        verb_classes: dict[int, str],
        noun_classes: dict[int, str],
        verb_instances: dict[int, list[str]] | None = None,
        noun_instances: dict[int, list[str]] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cuda",
    ):
        """
        Initialize the adapter.

        Args:
            verb_classes: Dict mapping verb_class (int) -> verb label (str)
            noun_classes: Dict mapping noun_class (int) -> noun label (str)
            verb_instances: Dict mapping verb_class (int) -> list of synonym
                strings from the EPIC_100_verb_classes.csv ``instances`` column.
                If *None*, falls back to using only the primary label.
            noun_instances: Dict mapping noun_class (int) -> list of instance
                strings from the EPIC_100_noun_classes.csv ``instances`` column.
                If *None*, falls back to using only the primary label.
            model_name: Sentence-transformer model for embedding
            device: Device for the embedding model
        """
        self.verb_classes = verb_classes
        self.noun_classes = noun_classes

        # Load sentence transformer
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is required for the adapter. "
                "Install with: pip install sentence-transformers"
            ) from err

        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

        # Sorted class id lists (stable iteration order)
        self._verb_ids = sorted(verb_classes.keys())
        self._noun_ids = sorted(noun_classes.keys())
        self._verb_labels = [verb_classes[i] for i in self._verb_ids]
        self._noun_labels = [noun_classes[i] for i in self._noun_ids]

        # Build multi-vector synonym embeddings
        self._verb_syn_embeddings = self._build_synonym_embeddings(
            self._verb_ids, verb_classes, verb_instances or {}
        )
        self._noun_syn_embeddings = self._build_synonym_embeddings(
            self._noun_ids, noun_classes, noun_instances or {}
        )

        total_verb_syns = sum(e.shape[0] for e in self._verb_syn_embeddings.values())
        total_noun_syns = sum(e.shape[0] for e in self._noun_syn_embeddings.values())
        logger.info(
            f"Multi-vector embeddings: {total_verb_syns} verb synonyms "
            f"across {len(self._verb_ids)} classes, "
            f"{total_noun_syns} noun instances "
            f"across {len(self._noun_ids)} classes"
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _build_synonym_embeddings(
        self,
        class_ids: list[int],
        class_labels: dict[int, str],
        class_instances: dict[int, list[str]],
    ) -> dict[int, np.ndarray]:
        """
        For each class, embed every synonym/instance and the primary label.

        Returns:
            Dict mapping class_id -> normalized embeddings array of shape
            ``(n_synonyms, embed_dim)`` where ``n_synonyms >= 1``.
        """
        syn_embeddings: dict[int, np.ndarray] = {}

        # Collect all texts to embed in one batch for efficiency
        all_texts: list[str] = []
        index_map: list[tuple[int, int, int]] = []  # (class_id, start, end)

        for cid in class_ids:
            label = class_labels[cid]
            # Gather synonyms: primary label + instances (de-duplicated)
            synonyms = [label]
            for inst in class_instances.get(cid, []):
                # EPIC instances use hyphens for multi-word (e.g. "pick-up")
                # and colons for compound nouns (e.g. "pan:frying").
                # Normalise colons to spaces so embedding captures semantics.
                normalised = inst.replace(":", " ").replace("-", " ").strip()
                if normalised and normalised not in synonyms:
                    synonyms.append(normalised)
                # Also keep the original hyphenated/colon form if different
                original = inst.strip()
                if original and original not in synonyms:
                    synonyms.append(original)

            start = len(all_texts)
            all_texts.extend(synonyms)
            end = len(all_texts)
            index_map.append((cid, start, end))

        logger.info(f"Encoding {len(all_texts)} synonym texts in batch...")
        all_embs = self.model.encode(
            all_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=256,
        )

        for cid, start, end in index_map:
            syn_embeddings[cid] = all_embs[start:end]

        return syn_embeddings

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _max_sim_match(
        self,
        query_emb: np.ndarray,
        syn_embeddings: dict[int, np.ndarray],
        class_ids: list[int],
        class_labels: list[str],
        top_k: int = 5,
    ) -> tuple[int, str, float, list[tuple[int, str, float]]]:
        """
        Multi-vector max-sim matching.

        For each class, compute cosine similarity between the query and
        **every** synonym embedding, take the max.  Return the class with
        the highest max similarity.

        Args:
            query_emb: Normalised query embedding of shape ``(1, dim)``
            syn_embeddings: Per-class synonym embeddings
            class_ids: Sorted class ID list
            class_labels: Corresponding canonical labels
            top_k: Number of top matches to return

        Returns:
            (best_class_id, best_label, best_similarity, top_k_list)
        """
        scores = np.empty(len(class_ids), dtype=np.float64)
        for idx, cid in enumerate(class_ids):
            embs = syn_embeddings[cid]  # (n_syn, dim)
            sims = (query_emb @ embs.T).squeeze()  # (n_syn,) or scalar
            scores[idx] = float(np.max(sims))

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_k_results = [(class_ids[i], class_labels[i], float(scores[i])) for i in top_indices]

        best_idx = int(top_indices[0])
        return (
            class_ids[best_idx],
            class_labels[best_idx],
            float(scores[best_idx]),
            top_k_results,
        )

    def map_verb(
        self,
        action_text: str,
        object_text: str = "",
        description_text: str = "",
        top_k: int = 5,
    ) -> tuple[int, str, float, list]:
        """
        Map a free-text action to the nearest EPIC verb class.

        Uses ``resolve_verb_text`` for alias resolution and context-aware
        disambiguation before embedding lookup.

        Args:
            action_text: Raw action string from VLM.
            object_text: Object string (for contextual disambiguation).
            description_text: Description string (for contextual disambiguation).
            top_k: Number of top matches to return.
        """
        text = resolve_verb_text(action_text, object_text, description_text)
        if not text:
            return self._verb_ids[0], self._verb_labels[0], 0.0, []

        query_emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self._max_sim_match(
            query_emb,
            self._verb_syn_embeddings,
            self._verb_ids,
            self._verb_labels,
            top_k,
        )

    def map_noun(
        self,
        object_text: str,
        top_k: int = 5,
    ) -> tuple[int, str, float, list]:
        """
        Map a free-text object description to the nearest EPIC noun class.

        Applies text preprocessing: splits compound descriptions and strips
        colour/material adjectives, then picks the candidate phrase that
        yields the highest max-sim score.
        """
        candidates = extract_noun_candidates(object_text)
        if not candidates:
            return self._noun_ids[0], self._noun_labels[0], 0.0, []

        # Embed all candidates at once
        cand_embs = self.model.encode(
            candidates,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        best_result: tuple[int, str, float, list] | None = None
        for emb in cand_embs:
            query_emb = emb.reshape(1, -1)
            result = self._max_sim_match(
                query_emb,
                self._noun_syn_embeddings,
                self._noun_ids,
                self._noun_labels,
                top_k,
            )
            if best_result is None or result[2] > best_result[2]:
                best_result = result

        return best_result  # type: ignore[return-value]

    def convert_prediction(
        self,
        video_id: str,
        clip_data: dict,
    ) -> MappedPrediction:
        """
        Convert a single ClipContext (as dict) to a MappedPrediction.

        Args:
            video_id: EPIC-KITCHENS video ID
            clip_data: Dict from clips_stage1.jsonl

        Returns:
            MappedPrediction with mapped class IDs
        """
        raw_action = clip_data.get("action", "")
        raw_object = clip_data.get("object", "")
        raw_description = clip_data.get("description", "")

        # Map verb: use action text, fall back to description
        # Pass object + description for context-aware disambiguation
        verb_text = raw_action if raw_action else raw_description
        verb_class, verb_label, verb_sim, verb_top5 = self.map_verb(
            verb_text,
            object_text=raw_object,
            description_text=raw_description,
        )

        # Map noun: use object text, fall back to description
        noun_text = raw_object if raw_object else raw_description
        noun_class, noun_label, noun_sim, noun_top5 = self.map_noun(noun_text)

        return MappedPrediction(
            video_id=video_id,
            start_t=clip_data.get("start_t", 0.0),
            end_t=clip_data.get("end_t", 0.0),
            raw_action=raw_action,
            raw_object=raw_object,
            raw_description=raw_description,
            verb_class=verb_class,
            noun_class=noun_class,
            verb_label=verb_label,
            noun_label=noun_label,
            verb_similarity=verb_sim,
            noun_similarity=noun_sim,
            verb_top5=verb_top5,
            noun_top5=noun_top5,
        )

    def convert_predictions(
        self,
        predictions_file: str | Path,
    ) -> list[MappedPrediction]:
        """
        Convert all predictions from an aggregated JSONL file.

        Args:
            predictions_file: Path to all_predictions.jsonl

        Returns:
            List of MappedPrediction objects
        """
        predictions_file = Path(predictions_file)
        mapped = []

        with open(predictions_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                clip_data = json.loads(line)

                # Use tagged video_id or derive from video_path
                video_id = clip_data.get("_video_id", "")
                if not video_id:
                    video_path = clip_data.get("video_path", "")
                    video_id = Path(video_path).stem if video_path else "unknown"

                pred = self.convert_prediction(video_id, clip_data)
                mapped.append(pred)

        logger.info(f"Converted {len(mapped)} predictions from {predictions_file}")
        return mapped

    def to_c2_submission(
        self,
        predictions: list[MappedPrediction],
        output_path: str | Path | None = None,
    ) -> dict:
        """
        Convert mapped predictions to C2-Action-Detection official submission format.

        The official C2 EvaluationCode expects:
        {
            "version": "0.2",
            "challenge": "action_detection",
            "sls_pt": 0, "sls_tl": 0, "sls_td": 0,
            "results": {
                "video_id": [
                    {
                        "segment": [start, end],
                        "verb": int,
                        "noun": int,
                        "action": "verb,noun",  # comma-separated string!
                        "score": float
                    }, ...
                ]
            }
        }

        Args:
            predictions: List of MappedPrediction objects
            output_path: Optional path to save submission JSON

        Returns:
            C2 submission dict
        """
        # Group by video
        by_video: dict[str, list] = {}
        for pred in predictions:
            entry = {
                "segment": [pred.start_t, pred.end_t],
                "verb": pred.verb_class,
                "noun": pred.noun_class,
                "action": f"{pred.verb_class},{pred.noun_class}",
                "score": pred.score,
            }
            by_video.setdefault(pred.video_id, []).append(entry)

        c2_submission = {
            "version": "0.2",
            "challenge": "action_detection",
            "sls_pt": 0,  # no pretraining on EPIC
            "sls_tl": 0,  # no training labels used
            "sls_td": 0,  # no training data used
            "results": by_video,
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(c2_submission, f, indent=2)

            logger.info(f"C2 submission saved to {output_path}")

        return c2_submission

    def save_mapped_predictions(
        self,
        predictions: list[MappedPrediction],
        output_path: str | Path,
    ) -> None:
        """Save mapped predictions as JSONL for downstream evaluation."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for pred in predictions:
                entry = {
                    "video_id": pred.video_id,
                    "start_t": pred.start_t,
                    "end_t": pred.end_t,
                    "raw_action": pred.raw_action,
                    "raw_object": pred.raw_object,
                    "raw_description": pred.raw_description,
                    "verb_class": pred.verb_class,
                    "noun_class": pred.noun_class,
                    "verb_label": pred.verb_label,
                    "noun_label": pred.noun_label,
                    "verb_similarity": pred.verb_similarity,
                    "noun_similarity": pred.noun_similarity,
                    "verb_top5": pred.verb_top5,
                    "noun_top5": pred.noun_top5,
                    "action_class": pred.action_class,
                    "score": pred.score,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(predictions)} mapped predictions to {output_path}")
