# Add an Embodiment

Add a contract and adapters; do not add robot-name branches to generic orchestration.

## Required implementation

1. Register a record task with a semantic `record` observation group.
2. Define state terms, transforms, units, and exact slice order.
3. Define absolute action terms and exact flat order.
4. Expose every training camera as a named semantic observation.
5. Add an `EmbodimentContract`; the generic converter must reject mixed or ambiguous recordings.
6. Add a GR00T modality config under `groot_finetune/`.
7. Add a registered closed-loop adapter mapping record observations to modality keys.
8. Register a record route and free-running inference task.
9. Use timeout-only source eligibility and define the task-success evaluator independently.
10. Define reset, render-only warmup, visual randomization, separate source/evaluation
    terminations, horizon, and FPS.
11. Measure camera-free collection and rendered evaluation capacities separately.

## Generic runner boundary

The transport and action adapter must remain embodiment-independent. The runner may interpret
declarative warmup/evaluation strategies or call registered hooks, but it must not compare a
contract ID to a robot literal.

If an embodiment needs behavior not expressible by the current adapter interface:

1. extend the adapter/specification interface generically;
2. implement it for existing contracts;
3. add synthetic tests proving the core does not depend on the new robot name;
4. then register the new embodiment.

## Contract tests

Test without IsaacLab where possible:

- state and action key order and dimensions;
- modality key agreement;
- transforms and inverse/normalization behavior;
- task evaluator sequences and per-environment reset;
- explicit converter contract validation and mixed-contract rejection;
- exact successful-episode filtering;
- frame-count audit failures.

Then run GPU integration checks:

- construct record and inference tasks;
- record or replay a short semantic episode;
- validate every camera;
- convert a small dataset;
- run an open-loop evaluation;
- run a full-horizon one-episode closed-loop smoke;
- capture one task-successful recording when the policy permits.

Document the new contract in one direct reference from `SKILL.md`. Keep detailed embodiment
knowledge out of the generic workflow.
