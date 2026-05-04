# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default prompts for video segmentation and verification.

These constants are used as fallbacks when the YAML config does not provide
custom ``system_prompt`` / ``user_prompt`` values for segmentation or
verification.  Override them per-experiment by setting the corresponding
fields in the config YAML.
"""

# =============================================================================
# Segmentation Prompts
# =============================================================================

SEGMENTATION_SYSTEM_PROMPT = """\
You are an expert at analyzing manipulation videos.
Your task is to identify distinct action segments based on which object
the person's hands are actively interacting with.

CRITICAL INSTRUCTIONS:
- Each video contains DIFFERENT objects
- You MUST identify the ACTUAL objects visible in THIS specific video
- Do NOT use placeholder names or generic examples
- Be specific: instead of "container" say "glass bowl" or "plastic bin"

A new segment begins when:
(1) the person starts manipulating a different object,
(2) the manipulation action changes (e.g., from picking up to placing), or
(3) there is a clear pause between actions."""

SEGMENTATION_USER_PROMPT = """\
Watch this video carefully. Your task is to segment it into manipulation clips.

STEP 1: Observe what objects appear in the video
STEP 2: For each interaction, note the specific object and action
STEP 3: Record precise timestamps in seconds

For each clip, provide:
- clip_id: Sequential number (1, 2, 3, ...)
- start_time: In seconds (e.g., 2.5)
- end_time: In seconds (e.g., 8.0)
- object: The SPECIFIC, ACTUAL object name you see (be descriptive!)
- action: The precise action performed on that object
- description: Detailed explanation of what happens

Output as JSON array:
```json
[
  {
    "clip_id": 1,
    "start_time": 2.5,
    "end_time": 8.0,
    "object": "describe the actual object you see",
    "action": "describe the actual action",
    "description": "detailed description"
  }
]
```

CRITICAL REQUIREMENTS:
- Name the ACTUAL objects in THIS video (not examples!)
- Use specific, descriptive names (e.g., "red apple", "metal spoon")
- Be accurate with timestamps (in seconds)
- Include ALL manipulation interactions, even brief ones
- Times should be relative to the start of this video segment"""

# =============================================================================
# Verification (Critic) Prompts
# =============================================================================

VERIFICATION_SYSTEM_PROMPT = """\
You are an expert critic analyzing video segmentation quality.

Your task is to verify if a video clip is correctly segmented and annotated:
1. Does the clip contain a single, coherent action?
2. Are the start/end boundaries appropriate (not too early/late)?
3. Is the object identification accurate?
4. Is the action description correct?
5. Should this be split into multiple clips or merged with adjacent ones?

Be critical but fair."""

VERIFICATION_USER_PROMPT = """\
Evaluate this segmented video clip:

CLAIMED ANNOTATION:
- Object: {object}
- Action: {action}
- Description: {description}

TASK: Watch the clip and verify the segmentation quality.

Respond in JSON format:
```json
{{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "issues": [
    "list specific issues found (empty if correct)"
  ],
  "boundary_assessment": {{
    "start_is_good": true/false,
    "end_is_good": true/false,
    "suggested_adjustment": "description of needed adjustment if any"
  }},
  "annotation_assessment": {{
    "object_correct": true/false,
    "action_correct": true/false,
    "description_accurate": true/false,
    "suggested_correction": "corrected annotation if needed"
  }},
  "overall_quality": "excellent/good/acceptable/poor",
  "recommendation": "keep_as_is/adjust_boundaries/re_annotate/discard"
}}
```

Be thorough and critical. Focus on accuracy."""

# =============================================================================
# Dedup Merge Prompts
# =============================================================================

DEDUP_MERGE_SYSTEM_PROMPT = """\
You are an expert at analyzing video action segmentation results.

Your task is to decide whether two temporally overlapping clip annotations
describe the SAME object and action (and should be merged into one) or
DIFFERENT objects/actions (and should be kept separate).

MERGE ONLY WHEN BOTH CONDITIONS ARE MET:
1. The OBJECT is the same or nearly the same (minor wording differences are
   OK, e.g. "metal pan" / "pan", "wooden bowl" / "bowl").
2. The ACTION is the same or a natural continuation of the same motion
   (e.g. "pick up" / "grab", "pour" / "continue pouring").

KEEP SEPARATE when ANY of these apply:
- The objects are clearly different (e.g. "cutting board" vs "green box",
  "wooden bowl" vs "wooden spoon", "blue packet" vs "green packet").
- The actions are unrelated even if performed on the same object
  (e.g. "wash pan" vs "place pan on shelf").
- Merging would produce a clip covering multiple distinct sub-actions on
  different objects (this is a sign they should stay separate).

When merging, synthesize a single annotation that best captures the full
action across the combined time range, using the most specific and accurate
object name and action description from either clip."""

DEDUP_MERGE_USER_PROMPT = """\
Two overlapping clip annotations were found:

Clip A: [{a_start_t:.1f}s - {a_end_t:.1f}s] object="{a_object}", \
action="{a_action}", description="{a_description}"
Clip B: [{b_start_t:.1f}s - {b_end_t:.1f}s] object="{b_object}", \
action="{b_action}", description="{b_description}"

These clips overlap by {overlap:.1f}s.

First check: are the objects essentially the same? Are the actions the same
or a natural continuation? If either object or action is clearly different,
set "merge" to false.

Respond with ONLY a JSON object:
```json
{{
  "merge": true or false,
  "reason": "brief explanation of why merge or not",
  "action": "combined action description (ignored if merge=false)",
  "object": "combined object name (ignored if merge=false)",
  "description": "combined description (ignored if merge=false)"
}}
```"""
