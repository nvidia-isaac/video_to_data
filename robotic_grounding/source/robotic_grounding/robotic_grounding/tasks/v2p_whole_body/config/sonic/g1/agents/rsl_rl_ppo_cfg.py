from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1SonicRslRlPpoCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for the Sonic G1 robotic grounding environment."""

    num_steps_per_env = 24
    max_iterations = 100_000
    save_interval = 500
    experiment_name = "g1_sonic_grounding"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.013,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.1,
        # rnd_cfg=RslRlRndCfg(
        #     weight=200.0,
        #     reward_normalization=True,
        #     state_normalization=True,
        #     predictor_hidden_dims=[64, 64],
        #     target_hidden_dims=[64, 64],
        # )
    )
