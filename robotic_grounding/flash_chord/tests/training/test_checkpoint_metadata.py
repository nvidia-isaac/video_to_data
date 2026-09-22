# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for required current checkpoint metadata and exact runtime schemas."""

from copy import deepcopy

import pytest


class _Action:
    input_mode = "raw"
    input_mapping = "identity"
    action_dim = 5
    block_names = ("wrist", "fingers")
    block_ranges = ((0, 3), (3, 5))


class _NormalizedAction(_Action):
    input_mode = "normalized"
    input_mapping = "linear"


class _RationalAction(_NormalizedAction):
    input_mapping = "rational"


class _Observation:
    observation_dim = 7
    block_names = ("state", "command")
    block_ranges = ((0, 4), (4, 7))


def test_policy_schema_records_exact_semantic_blocks_and_validates_compatibility():
    from flash_chord.training.checkpoint_metadata import policy_schema, validate_policy_schema

    schema = policy_schema(_Action(), _Observation())

    assert schema == {
        "action": {
            "input_mode": "raw",
            "input_mapping": "identity",
            "dimension": 5,
            "blocks": [
                {"name": "wrist", "start": 0, "end": 3},
                {"name": "fingers", "start": 3, "end": 5},
            ],
        },
        "observation": {
            "dimension": 7,
            "blocks": [
                {"name": "state", "start": 0, "end": 4},
                {"name": "command", "start": 4, "end": 7},
            ],
        },
    }
    validate_policy_schema(schema, schema)


@pytest.mark.parametrize("version", [1, 2, 3, 4, None])
def test_policy_schema_ignores_optional_version_labels_when_explicit_semantics_match(version):
    from flash_chord.training.checkpoint_metadata import policy_schema, validate_policy_schema

    current = policy_schema(_Action(), _Observation())
    labelled = {**current, "version": version}

    validate_policy_schema(labelled, current)
    validate_policy_schema(current, labelled)


def test_policy_schema_rejects_action_mapping_or_layout_changes():
    from flash_chord.training.checkpoint_metadata import policy_schema, validate_policy_schema

    linear = policy_schema(_NormalizedAction(), _Observation())
    rational = policy_schema(_RationalAction(), _Observation())
    with pytest.raises(ValueError, match="policy schema does not match"):
        validate_policy_schema(linear, rational)

    invalid_layout = deepcopy(linear)
    invalid_layout["action"]["blocks"][1]["start"] = 4
    with pytest.raises(ValueError, match="policy schema does not match"):
        validate_policy_schema(invalid_layout, linear)

    incomplete = deepcopy(linear)
    del incomplete["action"]["input_mapping"]
    with pytest.raises(ValueError, match="requires raw/identity"):
        validate_policy_schema(incomplete, linear)


@pytest.mark.parametrize(
    ("input_mode", "input_mapping"),
    (("raw", "linear"), ("raw", "rational"), ("normalized", "identity"), ("normalized", "unknown")),
)
def test_policy_schema_rejects_invalid_runtime_action_semantics(input_mode, input_mapping):
    from flash_chord.training.checkpoint_metadata import policy_schema

    action = _Action()
    action.input_mode = input_mode
    action.input_mapping = input_mapping
    with pytest.raises(ValueError, match="requires raw/identity"):
        policy_schema(action, _Observation())


def test_policy_schema_rejects_gaps_and_incomplete_coverage():
    from flash_chord.training.checkpoint_metadata import policy_schema

    class GappedAction(_Action):
        block_ranges = ((0, 2), (3, 5))

    class IncompleteObservation(_Observation):
        block_ranges = ((0, 4), (4, 6))

    with pytest.raises(ValueError, match="contiguous"):
        policy_schema(GappedAction(), _Observation())
    with pytest.raises(ValueError, match="expected dimension 7"):
        policy_schema(_Action(), IncompleteObservation())


def test_checkpoint_metadata_round_trips_and_requires_all_requested_fields():
    from flash_chord.training.checkpoint_metadata import (
        build_checkpoint_metadata,
        checkpoint_config,
        checkpoint_critic_schema,
        checkpoint_policy_schema,
        critic_schema,
        policy_schema,
    )

    config = {"task": {"parquet": "/data/reference.parquet"}, "seed": 42}
    policy = policy_schema(_Action(), _Observation())
    critic = critic_schema(_Action(), _Observation(), ("applied_voc", "target_voc"))
    metadata = build_checkpoint_metadata(config, policy, critic)

    assert checkpoint_config(metadata) == config
    assert checkpoint_policy_schema(metadata) == policy
    assert checkpoint_critic_schema(metadata) == critic
    for reader in (checkpoint_config, checkpoint_policy_schema, checkpoint_critic_schema):
        with pytest.raises(ValueError, match="required current-schema metadata"):
            reader({})


def test_checkpoint_metadata_rejects_malformed_json_and_nonmapping_values():
    from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY, checkpoint_config

    with pytest.raises(ValueError, match="invalid JSON"):
        checkpoint_config({CONFIG_METADATA_KEY: "{"})
    with pytest.raises(TypeError, match="must contain a mapping"):
        checkpoint_config({CONFIG_METADATA_KEY: "[]"})


def test_reference_config_requires_and_validates_current_articulation_contract():
    from flash_chord.data.reference import (
        ObjectArticulationSpec,
        ObjectAssetSpec,
        ObjectBodySpec,
        ObjectJointDriveSpec,
        ObjectJointPhysicsSpec,
    )
    from flash_chord.training.checkpoint_metadata import resolved_reference_config, validate_reference_config

    asset = ObjectAssetSpec(
        name="mixer",
        urdf_path="/assets/mixer.urdf",
        bodies=(ObjectBodySpec(reference_name="mixer", simulation_name="base"),),
        root_reference_name="mixer",
        articulations=(
            ObjectArticulationSpec(
                simulation_joint_name="mixer_joint",
                reference_index=0,
                physics=ObjectJointPhysicsSpec(armature=0.01, friction=0.1),
                drive=ObjectJointDriveSpec(kp=50.0, kd=2.0, effort_limit=50.0),
            ),
        ),
    )

    class Reference:
        num_frames = 849
        fps = 10.0

        def object_assets(self):
            return (asset,)

    reference = Reference()
    resolved = resolved_reference_config(reference, "/data/mixer.parquet")
    validate_reference_config({"reference": resolved}, reference)

    incompatible = deepcopy(resolved)
    incompatible["object_articulations"]["entries"][0]["physics"]["friction"] = 0.0
    with pytest.raises(ValueError, match="object articulation config does not match"):
        validate_reference_config({"reference": incompatible}, reference)
    with pytest.raises(ValueError, match="missing the object articulation contract"):
        validate_reference_config({"reference": {"parquet": "/old.parquet"}}, reference)
    with pytest.raises(ValueError, match="missing current reference metadata"):
        validate_reference_config({}, reference)


def test_resume_reference_config_requires_current_metadata_on_both_sides():
    from flash_chord.training.checkpoint_metadata import (
        build_checkpoint_metadata,
        policy_schema,
        validate_resume_reference_config,
    )

    config = {"reference": {"object_articulations": {"version": 1, "entries": []}}}
    metadata = build_checkpoint_metadata(config, policy_schema(_Action(), _Observation()))
    validate_resume_reference_config(metadata, metadata)
    with pytest.raises(ValueError, match="required current-schema metadata"):
        validate_resume_reference_config(metadata, None)


def test_critic_schema_records_context_order_rejects_reordering_and_ignores_version_labels():
    from flash_chord.training.checkpoint_metadata import critic_schema, validate_critic_schema

    saved = critic_schema(_Action(), _Observation(), ("applied_voc", "target_voc", "settling_progress"))
    assert saved["input_dimension"] == 15
    assert [block["name"] for block in saved["context"]["blocks"]] == [
        "applied_voc",
        "target_voc",
        "settling_progress",
    ]

    reordered = critic_schema(_Action(), _Observation(), ("target_voc", "applied_voc", "settling_progress"))
    with pytest.raises(ValueError, match="critic schema"):
        validate_critic_schema(saved, reordered)
    validate_critic_schema({**saved, "version": 2}, saved)


def test_checkpoint_evaluation_config_uses_saved_training_state_and_explicit_overrides():
    from flash_chord.configuration import compose_config, resolved_dict
    from flash_chord.training.checkpoint_metadata import checkpoint_evaluation_config

    saved = resolved_dict(
        compose_config(
            "train",
            ["experiment=sharpa_ppo", "task.parquet=/saved/reference.parquet", "scene.world_count=4096"],
        )
    )
    current = compose_config(
        "view_policy",
        [
            "task.parquet=/current/reference.parquet",
            "evaluation.checkpoint=/tmp/policy.ckpt",
            "scene.world_count=1",
            "viewer.backend=none",
        ],
    )
    effective = checkpoint_evaluation_config(
        current,
        saved,
        ("task.parquet", "evaluation.checkpoint", "viewer.backend"),
    )

    assert effective.task.parquet == "/current/reference.parquet"
    assert effective.evaluation.checkpoint == "/tmp/policy.ckpt"
    assert effective.viewer.backend == "none"
    assert effective.scene.world_count == effective.training.world_count == 1
    assert effective.logging.mode == "disabled"


def test_ppo_metadata_omits_critic_until_a_caller_requests_it():
    from flash_chord.training.checkpoint_metadata import (
        CRITIC_SCHEMA_METADATA_KEY,
        build_checkpoint_metadata,
        policy_schema,
    )

    metadata = build_checkpoint_metadata({"algorithm": "ppo"}, policy_schema(_Action(), _Observation()))
    assert CRITIC_SCHEMA_METADATA_KEY not in metadata
