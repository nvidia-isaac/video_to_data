# Closed-loop GR00T evaluation

Closed-loop evaluation binds a served checkpoint to an exact embodiment contract and task
adapter. The embodiment contract supplies state/action/camera mappings; the task profile supplies
the instruction, target object, and evaluator thresholds.

For Vega, use the `vega_sharpa_joint` contract with its registered
`VegaSharpa-WholeBody-Gr00t-Joint-Inference-v0` task. The evaluator supports the task-profile
selector `primary` or an exact scene-object name.

```bash
bash groot_finetune/closed_loop/run_eval.sh \
  --gr00t-dir /path/to/Isaac-GR00T \
  --model /path/to/checkpoint \
  --container robotic-grounding \
  --expected-mount-source /path/to/video_to_data \
  --client-workdir /workspace/video_to_data/robotic_grounding \
  --task VegaSharpa-WholeBody-Gr00t-Joint-Inference-v0 \
  --contract /workspace/run/contracts/embodiment.json \
  --task-profile /workspace/run/contracts/task_profile.json \
  --motion-file ego_recon/processed/sequence_id=example/robot_name=vega_sharpa \
  --human-motion-data-dir /workspace/human_motion_data \
  --expected-sequence-id example \
  --expected-robot-name vega_sharpa \
  --output-json /workspace/run/evaluation.json \
  --episodes 20 \
  --num-envs 4 \
  --episode-horizon 519
```

Evaluation keeps only contract-declared timeout and safety/non-finite terminations. The JSON
reports task success, termination counts, episode lengths, lift/hold measurements, contract and
task hashes, initialization metadata, and camera checks. Successful videos are filtered by the
same evaluator used for the reported success rate.
