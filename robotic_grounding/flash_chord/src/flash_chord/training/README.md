# Training package

PPO and FlashSAC are peer learners built on the same Warp/JAX environment boundary.

| Path | Responsibility |
| --- | --- |
| `environment.py` | Algorithm-neutral vector environment and transition types |
| `checkpoint.py` | Safetensors PyTree serialization and restoration |
| `checkpoint_metadata.py` | Saved config and exact policy, object-joint, and critic input descriptions |
| `ppo/` | PPO network, rollout, update, learner, runner, and evaluation |
| `flash_sac/` | FlashSAC network, replay, exploration, update, learner, runner, and evaluation |

Both learners index logs and checkpoints by cumulative control-step environment transitions. Physics substeps do
not advance this counter.

FlashSAC writes independent files:

| File | Contents |
| --- | --- |
| `policy_<steps>.safetensors` | Actor state and required current metadata for evaluation |
| `state_<steps>.safetensors` | Complete learner state for resume |
| `replay_<steps>.safetensors` | Optional replay snapshot |

A state-only resume starts with empty replay unless a matching replay snapshot is explicitly configured. Policy
files are evaluation artifacts and are not resumable training state.

All new checkpoints contain the complete saved configuration and explicit runtime input descriptions. Missing or
incompatible descriptions are rejected, and older configuration layouts are not migrated.

Run the focused tests with:

```bash
pytest tests/training tests/test_configuration.py -q
```

## Acknowledgements

### FlashSAC

Our JAX implementation is ported from [Holiday Robotics' FlashSAC](https://github.com/Holiday-Robot/FlashSAC),
by [Kim et al. (2026)](https://arxiv.org/abs/2604.04539). See the [upstream license notice](flash_sac/NOTICE).

```bibtex
@article{kim2026flashsac,
  title={{FlashSAC}: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control},
  author={Kim, Donghu and Lee, Youngdo and Park, Minho and Kim, Kinam and
          Nahendra, I Made Aswin and Seno, Takuma and Min, Sehee and Palenicek, Daniel and
          Vogt, Florian and Kragic, Danica and Peters, Jan and Choo, Jaegul and Lee, Hojoon},
  journal={arXiv preprint arXiv:2604.04539},
  year={2026},
  url={https://arxiv.org/abs/2604.04539}
}
```

### PPO

Our PPO implementation follows [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347).

```bibtex
@article{schulman2017proximal,
  title={Proximal Policy Optimization Algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017},
  url={https://arxiv.org/abs/1707.06347}
}
```
