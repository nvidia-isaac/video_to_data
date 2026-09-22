# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-resident uniform replay with upstream-compatible n-step aggregation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import math

import flax
import jax
import jax.numpy as jnp

from flash_chord.training.flash_sac.config import ReplayConfig


@dataclass(frozen=True)
class ReplaySpec:
    """Static replay layout and sampling behavior."""

    capacity: int
    minimum_size: int
    batch_size: int
    n_step: int
    discount: float
    world_count: int
    observation_dim: int
    action_dim: int
    objective_term_count: int
    critic_context_dim: int
    observation_storage_dtype: str = "float32"

    @classmethod
    def from_config(
        cls,
        config: ReplayConfig,
        *,
        discount: float,
        world_count: int,
        observation_dim: int,
        action_dim: int,
        objective_term_count: int,
        critic_context_dim: int,
    ) -> "ReplaySpec":
        """Combine Hydra replay settings with runtime policy dimensions."""
        return cls(
            capacity=config.capacity,
            minimum_size=config.minimum_size,
            batch_size=config.batch_size,
            n_step=config.n_step,
            discount=discount,
            world_count=world_count,
            observation_dim=observation_dim,
            action_dim=action_dim,
            objective_term_count=objective_term_count,
            critic_context_dim=critic_context_dim,
            observation_storage_dtype=config.observation_storage_dtype,
        )

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if not 0 < self.minimum_size <= self.capacity:
            raise ValueError(
                f"minimum_size must be in [1, capacity], got {self.minimum_size} for capacity {self.capacity}"
            )
        if not 0 < self.batch_size <= self.capacity:
            raise ValueError(f"batch_size must be in [1, capacity], got {self.batch_size} for capacity {self.capacity}")
        if self.n_step <= 0:
            raise ValueError(f"n_step must be positive, got {self.n_step}")
        if not math.isfinite(self.discount) or not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be finite and in (0, 1], got {self.discount}")
        dimensions = {
            "world_count": self.world_count,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "objective_term_count": self.objective_term_count,
            "critic_context_dim": self.critic_context_dim,
        }
        invalid_dimensions = {name: value for name, value in dimensions.items() if value <= 0}
        if invalid_dimensions:
            raise ValueError(f"replay dimensions must be positive, got {invalid_dimensions}")
        if self.capacity < self.world_count:
            raise ValueError(
                f"capacity {self.capacity} must hold at least one {self.world_count}-world transition batch"
            )
        if self.observation_storage_dtype not in {"float32", "float16"}:
            raise ValueError(
                f"observation_storage_dtype must be 'float32' or 'float16', got {self.observation_storage_dtype!r}"
            )

    @property
    def bootstrap_discount(self) -> float:
        """Fixed upstream n-step bootstrap multiplier, including shortened windows."""
        return self.discount**self.n_step


@dataclass(frozen=True)
class ReplayMemory:
    """Exact logical bytes for persistent and sampled replay arrays."""

    bytes_per_transition: int
    ring_bytes: int
    pending_bytes: int
    counter_bytes: int
    sampled_batch_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.ring_bytes + self.pending_bytes + self.counter_bytes


@flax.struct.dataclass
class ReplayInsert:
    """One vectorized environment interaction before n-step aggregation."""

    observation: jax.Array
    action: jax.Array
    objective_terms: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    next_observation: jax.Array
    critic_context: jax.Array
    next_critic_context: jax.Array


@flax.struct.dataclass
class ReplayStorage:
    """Committed ring payload arrays."""

    observation: jax.Array
    action: jax.Array
    discounted_objective_terms: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    next_observation: jax.Array
    critic_context: jax.Array
    next_critic_context: jax.Array


@flax.struct.dataclass
class PendingStorage:
    """Circular storage for the preceding ``n_step - 1`` vector interactions."""

    observation: jax.Array
    action: jax.Array
    objective_terms: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    next_observation: jax.Array
    critic_context: jax.Array
    next_critic_context: jax.Array
    size: jax.Array
    write_index: jax.Array


@flax.struct.dataclass
class ReplayState:
    """Complete immutable replay state; JIT insertion donates its payload buffers."""

    storage: ReplayStorage
    pending: PendingStorage
    size: jax.Array
    write_index: jax.Array
    spec: ReplaySpec = flax.struct.field(pytree_node=False)


@flax.struct.dataclass
class ReplayBatch:
    """One sampled batch of committed n-step transitions."""

    observation: jax.Array
    action: jax.Array
    discounted_objective_terms: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    next_observation: jax.Array
    critic_context: jax.Array
    next_critic_context: jax.Array


def replay_memory(spec: ReplaySpec) -> ReplayMemory:
    """Return exact logical array bytes without allocating replay storage."""
    observation_bytes = jnp.dtype(spec.observation_storage_dtype).itemsize
    float32_count = spec.action_dim + spec.objective_term_count + 2 * spec.critic_context_dim
    bytes_per_transition = 2 * spec.observation_dim * observation_bytes + 4 * float32_count + 2
    sampled_float_count = 2 * spec.observation_dim + float32_count
    pending_entries = (spec.n_step - 1) * spec.world_count
    counter_bytes = 4 * jnp.dtype(jnp.int32).itemsize
    return ReplayMemory(
        bytes_per_transition=bytes_per_transition,
        ring_bytes=spec.capacity * bytes_per_transition,
        pending_bytes=pending_entries * bytes_per_transition,
        counter_bytes=counter_bytes,
        sampled_batch_bytes=spec.batch_size * (4 * sampled_float_count + 2),
    )


def create_replay(spec: ReplaySpec, device: jax.Device | None = None) -> ReplayState:
    """Allocate an empty fixed-shape replay state on ``device``."""
    pending_count = spec.n_step - 1
    observation_dtype = jnp.dtype(spec.observation_storage_dtype)
    device_context = nullcontext() if device is None else jax.default_device(device)
    with device_context:
        # Keep the two capacity-dominant leaves adjacent in allocation order so smaller payloads cannot fragment
        # the device allocator between them.
        observation = jnp.zeros((spec.capacity, spec.observation_dim), dtype=observation_dtype)
        next_observation = jnp.zeros((spec.capacity, spec.observation_dim), dtype=observation_dtype)
        storage = ReplayStorage(
            observation=observation,
            action=jnp.zeros((spec.capacity, spec.action_dim), dtype=jnp.float32),
            discounted_objective_terms=jnp.zeros((spec.capacity, spec.objective_term_count), dtype=jnp.float32),
            terminated=jnp.zeros((spec.capacity,), dtype=jnp.bool_),
            truncated=jnp.zeros((spec.capacity,), dtype=jnp.bool_),
            next_observation=next_observation,
            critic_context=jnp.zeros((spec.capacity, spec.critic_context_dim), dtype=jnp.float32),
            next_critic_context=jnp.zeros((spec.capacity, spec.critic_context_dim), dtype=jnp.float32),
        )
        pending_shape = (pending_count, spec.world_count)
        pending = PendingStorage(
            observation=jnp.zeros((*pending_shape, spec.observation_dim), dtype=observation_dtype),
            action=jnp.zeros((*pending_shape, spec.action_dim), dtype=jnp.float32),
            objective_terms=jnp.zeros((*pending_shape, spec.objective_term_count), dtype=jnp.float32),
            terminated=jnp.zeros(pending_shape, dtype=jnp.bool_),
            truncated=jnp.zeros(pending_shape, dtype=jnp.bool_),
            next_observation=jnp.zeros((*pending_shape, spec.observation_dim), dtype=observation_dtype),
            critic_context=jnp.zeros((*pending_shape, spec.critic_context_dim), dtype=jnp.float32),
            next_critic_context=jnp.zeros((*pending_shape, spec.critic_context_dim), dtype=jnp.float32),
            size=jnp.asarray(0, dtype=jnp.int32),
            write_index=jnp.asarray(0, dtype=jnp.int32),
        )
        return ReplayState(
            storage=storage,
            pending=pending,
            size=jnp.asarray(0, dtype=jnp.int32),
            write_index=jnp.asarray(0, dtype=jnp.int32),
            spec=spec,
        )


def _validate_insert(state: ReplayState, transition: ReplayInsert) -> None:
    spec = state.spec
    expected = {
        "observation": ((spec.world_count, spec.observation_dim), jnp.float32),
        "action": ((spec.world_count, spec.action_dim), jnp.float32),
        "objective_terms": ((spec.world_count, spec.objective_term_count), jnp.float32),
        "terminated": ((spec.world_count,), jnp.bool_),
        "truncated": ((spec.world_count,), jnp.bool_),
        "next_observation": ((spec.world_count, spec.observation_dim), jnp.float32),
        "critic_context": ((spec.world_count, spec.critic_context_dim), jnp.float32),
        "next_critic_context": ((spec.world_count, spec.critic_context_dim), jnp.float32),
    }
    for name, (shape, dtype) in expected.items():
        value = getattr(transition, name)
        if value.shape != shape:
            raise ValueError(f"replay insert {name} has shape {value.shape}; expected {shape}")
        if value.dtype != jnp.dtype(dtype):
            raise TypeError(f"replay insert {name} has dtype {value.dtype}; expected {jnp.dtype(dtype)}")


def _write_pending(pending: PendingStorage, index: jax.Array, transition: ReplayInsert) -> PendingStorage:
    return pending.replace(
        observation=pending.observation.at[index].set(transition.observation.astype(pending.observation.dtype)),
        action=pending.action.at[index].set(transition.action),
        objective_terms=pending.objective_terms.at[index].set(transition.objective_terms),
        terminated=pending.terminated.at[index].set(transition.terminated),
        truncated=pending.truncated.at[index].set(transition.truncated),
        next_observation=pending.next_observation.at[index].set(
            transition.next_observation.astype(pending.next_observation.dtype)
        ),
        critic_context=pending.critic_context.at[index].set(transition.critic_context),
        next_critic_context=pending.next_critic_context.at[index].set(transition.next_critic_context),
    )


def _aggregate_pending(pending: PendingStorage, transition: ReplayInsert, spec: ReplaySpec) -> ReplayBatch:
    """Aggregate one full n-step window exactly like upstream FlashSAC."""
    discounted_terms = transition.objective_terms
    terminated = transition.terminated
    truncated = transition.truncated
    next_observation = transition.next_observation
    next_critic_context = transition.next_critic_context
    pending_count = spec.n_step - 1

    for offset in reversed(range(pending_count)):
        index = (pending.write_index + offset) % pending_count
        prior_terminated = pending.terminated[index]
        prior_truncated = pending.truncated[index]
        done = jnp.logical_or(prior_terminated, prior_truncated)
        discounted_terms = pending.objective_terms[index] + spec.discount * discounted_terms * (~done[:, None])
        terminated = jnp.where(done, prior_terminated, terminated)
        truncated = jnp.where(done, prior_truncated, truncated)
        next_observation = jnp.where(done[:, None], pending.next_observation[index], next_observation)
        next_critic_context = jnp.where(
            done[:, None],
            pending.next_critic_context[index],
            next_critic_context,
        )

    oldest = pending.write_index
    return ReplayBatch(
        observation=pending.observation[oldest],
        action=pending.action[oldest],
        discounted_objective_terms=discounted_terms,
        terminated=terminated,
        truncated=truncated,
        next_observation=next_observation,
        critic_context=pending.critic_context[oldest],
        next_critic_context=next_critic_context,
    )


def _one_step_batch(transition: ReplayInsert) -> ReplayBatch:
    return ReplayBatch(
        observation=transition.observation,
        action=transition.action,
        discounted_objective_terms=transition.objective_terms,
        terminated=transition.terminated,
        truncated=transition.truncated,
        next_observation=transition.next_observation,
        critic_context=transition.critic_context,
        next_critic_context=transition.next_critic_context,
    )


def _append_batch(state: ReplayState, batch: ReplayBatch) -> ReplayState:
    spec = state.spec
    indices = (state.write_index + jnp.arange(spec.world_count, dtype=jnp.int32)) % spec.capacity
    storage = state.storage.replace(
        observation=state.storage.observation.at[indices].set(
            batch.observation.astype(state.storage.observation.dtype)
        ),
        action=state.storage.action.at[indices].set(batch.action),
        discounted_objective_terms=state.storage.discounted_objective_terms.at[indices].set(
            batch.discounted_objective_terms
        ),
        terminated=state.storage.terminated.at[indices].set(batch.terminated),
        truncated=state.storage.truncated.at[indices].set(batch.truncated),
        next_observation=state.storage.next_observation.at[indices].set(
            batch.next_observation.astype(state.storage.next_observation.dtype)
        ),
        critic_context=state.storage.critic_context.at[indices].set(batch.critic_context),
        next_critic_context=state.storage.next_critic_context.at[indices].set(batch.next_critic_context),
    )
    return state.replace(
        storage=storage,
        size=jnp.minimum(state.size + spec.world_count, spec.capacity),
        write_index=(state.write_index + spec.world_count) % spec.capacity,
    )


def _insert_replay(state: ReplayState, transition: ReplayInsert) -> ReplayState:
    """Pure insertion implementation used by the composed learner executable."""
    _validate_insert(state, transition)
    pending_count = state.spec.n_step - 1
    if pending_count == 0:
        return _append_batch(state, _one_step_batch(transition))

    def store_pending(operand: tuple[ReplayState, ReplayInsert]) -> ReplayState:
        current, insert = operand
        pending = _write_pending(current.pending, current.pending.write_index, insert)
        pending = pending.replace(
            size=pending.size + 1,
            write_index=(pending.write_index + 1) % pending_count,
        )
        return current.replace(pending=pending)

    def commit_window(operand: tuple[ReplayState, ReplayInsert]) -> ReplayState:
        current, insert = operand
        batch = _aggregate_pending(current.pending, insert, current.spec)
        current = _append_batch(current, batch)
        pending = _write_pending(current.pending, current.pending.write_index, insert)
        pending = pending.replace(write_index=(pending.write_index + 1) % pending_count)
        return current.replace(pending=pending)

    return jax.lax.cond(
        state.pending.size < pending_count,
        store_pending,
        commit_window,
        (state, transition),
    )


@partial(jax.jit, donate_argnums=(0,))
def insert_replay(state: ReplayState, transition: ReplayInsert) -> ReplayState:
    """Snapshot one vector interaction and commit its oldest full n-step window."""
    return _insert_replay(state, transition)


def _gather(storage: ReplayStorage, indices: jax.Array) -> ReplayBatch:
    return ReplayBatch(
        observation=storage.observation[indices].astype(jnp.float32),
        action=storage.action[indices],
        discounted_objective_terms=storage.discounted_objective_terms[indices],
        terminated=storage.terminated[indices],
        truncated=storage.truncated[indices],
        next_observation=storage.next_observation[indices].astype(jnp.float32),
        critic_context=storage.critic_context[indices],
        next_critic_context=storage.next_critic_context[indices],
    )


@jax.jit
def gather_replay(state: ReplayState, indices: jax.Array) -> ReplayBatch:
    """Gather explicit physical ring indices for deterministic diagnostics and tests."""
    if indices.ndim != 1:
        raise ValueError(f"replay indices must be rank one, got {indices.shape}")
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise TypeError(f"replay indices must have integer dtype, got {indices.dtype}")
    return _gather(state.storage, indices)


def _sample_replay(state: ReplayState, key: jax.Array) -> ReplayBatch:
    """Pure sampling implementation used by the composed learner executable."""
    indices = jax.random.randint(
        key,
        (state.spec.batch_size,),
        minval=0,
        maxval=state.size,
        dtype=jnp.int32,
    )
    return _gather(state.storage, indices)


@jax.jit
def sample_replay(state: ReplayState, key: jax.Array) -> ReplayBatch:
    """Sample uniformly with replacement from committed entries."""
    return _sample_replay(state, key)


def _can_sample(state: ReplayState) -> jax.Array:
    """Pure replay-readiness predicate for the composed learner executable."""
    return state.size >= state.spec.minimum_size


@jax.jit
def can_sample(state: ReplayState) -> jax.Array:
    """Return whether the committed ring has reached its configured warm-up."""
    return _can_sample(state)


def _relabel_rewards(batch: ReplayBatch, objective_weights: jax.Array, frame_dt: jax.Array) -> jax.Array:
    """Pure current-weight reward relabeling for the composed learner executable."""
    if objective_weights.shape != (batch.discounted_objective_terms.shape[-1],):
        raise ValueError(
            f"objective_weights has shape {objective_weights.shape}; "
            f"expected {(batch.discounted_objective_terms.shape[-1],)}"
        )
    return frame_dt * jnp.einsum("bt,t->b", batch.discounted_objective_terms, objective_weights)


@jax.jit
def relabel_rewards(batch: ReplayBatch, objective_weights: jax.Array, frame_dt: jax.Array) -> jax.Array:
    """Apply current curriculum weights to stored discounted raw objective terms."""
    return _relabel_rewards(batch, objective_weights, frame_dt)


def _flush_pending(state: ReplayState) -> ReplayState:
    pending = state.pending.replace(
        size=jnp.asarray(0, dtype=jnp.int32),
        write_index=jnp.asarray(0, dtype=jnp.int32),
    )
    return state.replace(pending=pending)


@partial(jax.jit, donate_argnums=(0,))
def flush_pending(state: ReplayState) -> ReplayState:
    """Drop in-flight starts without changing committed FIFO entries or payload arrays."""
    return _flush_pending(state)


@partial(jax.jit, donate_argnums=(0,))
def clear_replay(state: ReplayState) -> ReplayState:
    """Logically clear committed and pending entries without zeroing payload arrays."""
    return _clear_replay(state)


def _clear_replay(state: ReplayState) -> ReplayState:
    """Pure logical clear used by the composed learner executable."""
    state = _flush_pending(state)
    return state.replace(
        size=jnp.asarray(0, dtype=jnp.int32),
        write_index=jnp.asarray(0, dtype=jnp.int32),
    )
