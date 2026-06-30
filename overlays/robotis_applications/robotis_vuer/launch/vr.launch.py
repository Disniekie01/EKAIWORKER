#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Wonho Yun

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='sh5',
        description='VR model to run: hx5, sg2, or sh5',
    )
    enable_lift_arg = DeclareLaunchArgument(
        'enable_lift_publishing',
        default_value='false',
        description='SH5 whole-body: publish lift_joint from headset height.',
    )
    enable_head_arg = DeclareLaunchArgument(
        'enable_head_publishing',
        default_value='false',
        description='SH5 whole-body: publish head joints from headset pose.',
    )
    enable_base_arg = DeclareLaunchArgument(
        'enable_base_publishing',
        default_value='false',
        description='SH5 whole-body: publish base cmd_vel from operator motion.',
    )
    enable_vr_image_arg = DeclareLaunchArgument(
        'enable_vr_image',
        default_value='false',
        description='Stream stereo passthrough image to the headset.',
    )
    lift_control_mode_arg = DeclareLaunchArgument(
        'lift_control_mode',
        default_value='head',
        description="SH5 lift control: 'head' (track headset height) or "
                    "'gesture' (both-hands pinch=up / squeeze=down, holds on release).",
    )
    lift_jog_velocity_arg = DeclareLaunchArgument(
        'lift_jog_velocity',
        default_value='0.08',
        description='SH5 gesture lift jog speed in m/s while a jog gesture is held.',
    )

    model = LaunchConfiguration('model')
    sh5_whole_body_params = [{
        'enable_lift_publishing': ParameterValue(
            LaunchConfiguration('enable_lift_publishing'), value_type=bool
        ),
        'enable_head_publishing': ParameterValue(
            LaunchConfiguration('enable_head_publishing'), value_type=bool
        ),
        'enable_base_publishing': ParameterValue(
            LaunchConfiguration('enable_base_publishing'), value_type=bool
        ),
        'enable_vr_image': ParameterValue(
            LaunchConfiguration('enable_vr_image'), value_type=bool
        ),
        'lift_control_mode': ParameterValue(
            LaunchConfiguration('lift_control_mode'), value_type=str
        ),
        'lift_jog_velocity': ParameterValue(
            LaunchConfiguration('lift_jog_velocity'), value_type=float
        ),
    }]
    sg2_node = Node(
        package='robotis_vuer',
        executable='vr_publisher_sg2',
        name='vr_publisher_sg2',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(
            PythonExpression(["'true' if '", model, "' == 'sg2' else 'false'"])
        ),
    )
    sh5_node = Node(
        package='robotis_vuer',
        executable='vr_publisher_sh5',
        name='vr_publisher_sh5',
        output='screen',
        emulate_tty=True,
        parameters=sh5_whole_body_params,
        condition=IfCondition(
            PythonExpression(["'true' if '", model, "' == 'sh5' else 'false'"])
        ),
    )
    hx5_node = Node(
        package='robotis_vuer',
        executable='vr_publisher_hx5',
        name='vr_publisher_hx5',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(
            PythonExpression(["'true' if '", model, "' == 'hx5' else 'false'"])
        ),
    )

    return LaunchDescription([
        model_arg,
        enable_lift_arg,
        enable_head_arg,
        enable_base_arg,
        enable_vr_image_arg,
        lift_control_mode_arg,
        lift_jog_velocity_arg,
        sg2_node,
        sh5_node,
        hx5_node,
    ])
