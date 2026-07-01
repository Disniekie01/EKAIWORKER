import gymnasium as gym

gym.register(
    id="Cyclo-Real-Box-Stack-FFW-SH5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FFWSH5BoxStackEnvCfg",
    },
    disable_env_checker=True,
)

_SH5_MIMIC_ENTRY = (
    "cyclo_lab.real_world_tasks.manager_based.FFW_SH5.sh5_mimic_env:FFWSH5DualArmMimicEnv"
)

gym.register(
    id="Cyclo-Real-Mimic-Box-Stack-FFW-SH5-v0",
    entry_point=_SH5_MIMIC_ENTRY,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.box_stack_mimic_env_cfg:FFWSH5BoxStackMimicEnvCfg",
    },
    disable_env_checker=True,
)
