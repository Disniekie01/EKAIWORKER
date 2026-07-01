import gymnasium as gym

gym.register(
    id="Cyclo-Real-Pick-Place-LTable-FFW-SH5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FFWSH5PickPlaceLTableEnvCfg",
    },
    disable_env_checker=True,
)

_SH5_MIMIC_ENTRY = (
    "cyclo_lab.real_world_tasks.manager_based.FFW_SH5.sh5_mimic_env:FFWSH5DualArmMimicEnv"
)

gym.register(
    id="Cyclo-Real-Mimic-Pick-Place-LTable-FFW-SH5-v0",
    entry_point=_SH5_MIMIC_ENTRY,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_l_table_mimic_env_cfg:FFWSH5PickPlaceLTableMimicEnvCfg",
    },
    disable_env_checker=True,
)
