# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulation timing + solver configuration (the single source of truth for physics knobs).

The backend is MuJoCo-Warp via Newton's ``SolverMuJoCo`` — there is no separate physics engine to
select; whether it runs on CPU or CUDA is decided by the device the model's Warp arrays live on.
``solver`` / ``integrator`` / ``cone`` are MuJoCo solver settings (the constraint solver algorithm,
time integrator, and friction cone), not engine choices. Both the replay and the RL/MPPI env consume
one :class:`SimConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import newton
import warp as wp


@dataclass
class SimConfig:
    """Physics timing + MuJoCo-Warp (``SolverMuJoCo``) settings."""

    fps: float = 50.0  # control rate [Hz]; one control frame = `substeps` physics steps
    substeps: int = 4  # physics substeps per control frame (decimation) -> 200 Hz physics
    # The reference is resampled to `fps`; the tracking gains are tuned for this rate.

    # SolverMuJoCo settings
    solver: str = "newton"  # constraint solver: "newton" | "cg"
    integrator: str = "implicitfast"  # "euler" | "implicit" | "implicitfast" | "rk4"
    cone: str = "pyramidal"  # friction cone: "elliptic" | "pyramidal"
    njmax: int = 16384  # max constraints / world (deep mesh grasps generate 1000s of contacts)
    nconmax: int = 4096  # max contacts / world
    impratio: float = 20.0
    iterations: int = 100
    ls_iterations: int = 50
    use_mujoco_contacts: bool = True  # True: MuJoCo native contacts (geom solref/solimp apply, needed for
    # grasping). False: Newton's Warp collision pipeline, which ignores the MuJoCo geom contact params.

    @property
    def frame_dt(self) -> float:
        """Control timestep [s] (``1/fps``)."""
        return 1.0 / self.fps

    @property
    def sim_dt(self) -> float:
        """Physics substep timestep [s] (``frame_dt / substeps``)."""
        return self.frame_dt / self.substeps

    @property
    def physics_fps(self) -> float:
        """Physics rate [Hz] (``fps * substeps``)."""
        return self.fps * self.substeps


def build_solver(model, cfg: SimConfig):
    """Construct the MuJoCo-Warp solver (Newton ``SolverMuJoCo``) for ``model`` from ``cfg``."""
    return newton.solvers.SolverMuJoCo(
        model,
        solver=cfg.solver,
        integrator=cfg.integrator,
        njmax=cfg.njmax,
        nconmax=cfg.nconmax,
        impratio=cfg.impratio,
        cone=cfg.cone,
        iterations=cfg.iterations,
        ls_iterations=cfg.ls_iterations,
        use_mujoco_contacts=cfg.use_mujoco_contacts,
    )


def build_force_contacts(model, solver, cfg: SimConfig, device=None):
    """Allocate the force-capable contact buffer required by one solver configuration."""
    if cfg.use_mujoco_contacts:
        contacts = newton.Contacts(
            rigid_contact_max=solver.get_max_contact_count(),
            soft_contact_max=0,
            device=wp.get_device(device),
            requested_attributes={"force"},
        )
    else:
        contacts = model.contacts()
    if contacts.force is None:
        raise ValueError("scene contacts do not include the extended force attribute")
    return contacts
