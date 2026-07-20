import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


EXPERIMENTS = {
    'baseline': ('params_baseline.yaml', 'none'),
    'pseudo': ('params_pseudo_semantic.yaml', 'pseudo'),
    'true': ('params_true_semantic.yaml', 'replay'),
}


def _launch_experiment(context):
    experiment = LaunchConfiguration('experiment').perform(context)
    if experiment not in EXPERIMENTS:
        choices = ', '.join(EXPERIMENTS)
        raise RuntimeError(f'Unknown experiment {experiment!r}; expected one of: {choices}')

    semantic_dir = LaunchConfiguration('semantic_dir').perform(context)
    if experiment == 'true' and not semantic_dir:
        raise RuntimeError('true experiment requires semantic_dir:=<precomputed Cylinder3D frames>')

    share_dir = get_package_share_directory('dsw_lio_sam')
    params_name, semantic_source = EXPERIMENTS[experiment]
    run_launch = os.path.join(share_dir, 'launch', 'run.launch.py')
    params_file = os.path.join(share_dir, 'config', params_name)

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(run_launch),
            launch_arguments={
                'params_file': params_file,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'start_pseudo_semantic_bridge': 'true' if semantic_source == 'pseudo' else 'false',
            }.items(),
        )
    ]
    if semantic_source == 'replay':
        actions.append(
            Node(
                package='dsw_lio_sam',
                executable='semantic_replay_node',
                name='semantic_replay_node',
                parameters=[{
                    'semantic_dir': semantic_dir,
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                }],
                output='screen',
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'experiment',
            default_value='baseline',
            description='Experiment mode: baseline, pseudo, or true'),
        DeclareLaunchArgument(
            'semantic_dir',
            default_value='',
            description='Precomputed Cylinder3D semantic frames (required for true mode)'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use rosbag /clock'),
        OpaqueFunction(function=_launch_experiment),
    ])
