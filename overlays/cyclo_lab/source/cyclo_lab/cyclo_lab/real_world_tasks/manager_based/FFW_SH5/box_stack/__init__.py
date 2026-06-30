import gymnasium as gym

gym.register(
    id="Cyclo-Real-Box-Stack-FFW-SH5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FFWSH5BoxStackEnvCfg",
    },
    disable_env_checker=True,
)
