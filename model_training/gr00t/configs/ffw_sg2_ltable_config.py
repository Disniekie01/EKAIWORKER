# SPDX-License-Identifier: Apache-2.0
#
# FFW_SG2 L-Table modality config for GR00T N1.7 fine-tuning.
# Derived from ROBOTIS's examples/CYCLO/ffw_sg2_rev1/ffw_sg2_rev1_config.py,
# trimmed to match JSHNSL/ffw_sg2_ltable_merge:
#   - 19-dim state/action  (odometry removed)
#   - single camera        (cam_left_head only; wrist cams removed)
#
# Place at: <Isaac-GR00T>/examples/CYCLO/ffw_sg2_ltable/ffw_sg2_ltable_config.py

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


_ACTION_KEYS = [
    "arm_left",
    "arm_right",
    "head",
    "lift",
]


ffw_sg2_ltable_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_left_head",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm_left",
            "arm_right",
            "head",
            "lift",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=_ACTION_KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in _ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.primitive_instruction"],
    ),
}


register_modality_config(ffw_sg2_ltable_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
