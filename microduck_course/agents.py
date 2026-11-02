"""PPO cho tầng chiến thuật đá bóng (đầu vào ~19 số, đầu ra 3 lệnh vận tốc)."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class SoccerScorePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32  # 3,2 s mỗi lần thu thập ở 10 Hz
    max_iterations = 1500
    save_interval = 100
    experiment_name = "microduck_soccer_score"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
