"""MicroDuck với vật dụng NVIDIA SimReady: sân "bàn làm việc tí hon", sân bóng (để xem, dùng chính sách đi bộ)
và bài RL học ghi bàn (tầng chiến thuật trên chính sách đi bộ)."""

import gymnasium as gym

_AGENT = "isaaclab_tasks.contrib.microduck.agents.rsl_rl_ppo_cfg:MicroDuckPPORunnerCfg"

gym.register(
    id="IsaacContrib-Velocity-Course-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.course_env_cfg:MicroDuckCourseEnvCfg",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)

gym.register(
    id="IsaacContrib-Soccer-Score-MicroDuck",
    entry_point=f"{__name__}.soccer_rl:HierarchicalRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.soccer_score_env_cfg:MicroDuckSoccerScoreEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:SoccerScorePPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Soccer-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.soccer_env_cfg:MicroDuckSoccerEnvCfg",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
)

gym.register(
    id="IsaacContrib-Soccer-Team-MicroDuck",
    entry_point=f"{__name__}.soccer_rl:HierarchicalRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.soccer_team:MicroDuckSoccerTeamEnvCfg"},
)
