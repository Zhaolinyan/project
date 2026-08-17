import os
import fcntl
import glob
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


EXPERIMENTS = {
    'original': ('params_original_lio_sam.yaml', 'none'),
    'baseline': ('params_baseline.yaml', 'none'),
    'pseudo': ('params_pseudo_semantic.yaml', 'pseudo'),
    'true': ('params_true_semantic.yaml', 'replay'),
}
_EXPERIMENT_LOCKS = {}


def _find_existing_pipeline_processes():
    executable_names = (
        'dsw_lio_sam_imuPreintegration',
        'dsw_lio_sam_imageProjection',
        'dsw_lio_sam_featureExtraction',
        'dsw_lio_sam_mapOptimization',
        'dsw_lio_sam_semanticBridge',
        'semantic_replay_node',
    )
    existing = []
    for command_path in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            with open(command_path, 'rb') as command_file:
                command = command_file.read().replace(b'\0', b' ').decode(
                    'utf-8', errors='replace')
        except (OSError, PermissionError):
            continue
        if any(name in command for name in executable_names):
            existing.append(f'{command_path.split("/")[2]}:{command.strip()}')
    return existing


def _acquire_experiment_lock(experiment):
    lock_path = os.path.join(
        tempfile.gettempdir(), 'dsw_lio_sam_experiment.lock')
    lock_file = open(lock_path, 'w', encoding='ascii')
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(
            'A DSW-LIO-SAM experiment launch is already running; stop it '
            f'before starting {experiment}') from exc
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    _EXPERIMENT_LOCKS[experiment] = lock_file


def _launch_experiment(context):
    experiment = LaunchConfiguration('experiment').perform(context)
    if experiment not in EXPERIMENTS:
        choices = ', '.join(EXPERIMENTS)
        raise RuntimeError(f'Unknown experiment {experiment!r}; expected one of: {choices}')
    _acquire_experiment_lock(experiment)
    existing_processes = _find_existing_pipeline_processes()
    if existing_processes:
        details = '\n  '.join(existing_processes[:8])
        raise RuntimeError(
            'Existing DSW-LIO-SAM node processes would contaminate this '
            f'experiment:\n  {details}')

    semantic_dir = LaunchConfiguration('semantic_dir').perform(context)
    if experiment == 'true' and not semantic_dir:
        raise RuntimeError(
            'true experiment requires '
            'semantic_dir:=<precomputed Cylinder3D frames>')

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
                'start_rviz': LaunchConfiguration('start_rviz'),
            }.items(),
        )
    ]
    if semantic_source == 'replay':
        semantic_replay_node = Node(
            package='dsw_lio_sam',
            executable='semantic_replay_node',
            name='semantic_replay_node',
            parameters=[{
                'semantic_dir': semantic_dir,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
            output='screen',
            on_exit=EmitEvent(event=Shutdown(
                reason='semantic replay node stopped; aborting true semantic experiment')),
        )
        actions.append(semantic_replay_node)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'experiment',
            default_value='baseline',
            description='Experiment mode: original, baseline, pseudo, or true'),
        DeclareLaunchArgument(
            'semantic_dir',
            default_value='',
            description='Precomputed Cylinder3D semantic frames (required for true mode)'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use rosbag /clock'),
        DeclareLaunchArgument(
            'start_rviz',
            default_value='true',
            description='Start RViz when true'),
        OpaqueFunction(function=_launch_experiment),
    ])
