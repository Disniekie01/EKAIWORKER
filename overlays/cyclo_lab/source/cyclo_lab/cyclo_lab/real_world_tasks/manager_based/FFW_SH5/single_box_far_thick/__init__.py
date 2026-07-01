import gymnasium as gym

gym.register(
    id="Cyclo-Real-Single-Box-Far-Thick-FFW-SH5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FFWSH5SingleBoxFarThickEnvCfg",
    },
    disable_env_checker=True,
)

_SH5_MIMIC_ENTRY = (
    "cyclo_lab.real_world_tasks.manager_based.FFW_SH5.sh5_mimic_env:FFWSH5DualArmMimicEnv"
)

gym.register(
    id="Cyclo-Real-Mimic-Single-Box-Far-Thick-FFW-SH5-v0",
    entry_point=_SH5_MIMIC_ENTRY,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.single_box_far_thick_mimic_env_cfg:FFWSH5SingleBoxFarThickMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
