import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    share_dir = get_package_share_directory('dsw_lio_sam')
    parameter_file = LaunchConfiguration('params_file')
    xacro_path = os.path.join(share_dir, 'config', 'robot.urdf.xacro')
    rviz_config_file = os.path.join(share_dir, 'config', 'rviz2.rviz')

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            share_dir, 'config', 'params.yaml'),
        description='FPath to the ROS2 parameters file to use.')

    use_sim_time_declare = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (bag) time if true')

    semantic_enabled_declare = DeclareLaunchArgument(
        'semantic_enabled',
        default_value='false',
        description='Deprecated compatibility arg; does not start a semantic publisher')

    start_pseudo_semantic_bridge_declare = DeclareLaunchArgument(
        'start_pseudo_semantic_bridge',
        default_value='false',
        description='Start the built-in geometry-rule pseudo semantic bridge if true')

    start_rviz_declare = DeclareLaunchArgument(
        'start_rviz',
        default_value='true',
        description='Start RViz when true')

    print("urdf_file_name : {}".format(xacro_path))

    return LaunchDescription([
        params_declare,
        use_sim_time_declare,
        semantic_enabled_declare,
        start_pseudo_semantic_bridge_declare,
        start_rviz_declare,

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments='0.0 0.0 0.0 0.0 0.0 0.0 map odom'.split(' '),
            parameters=[parameter_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen'
            ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': Command(['xacro', ' ', xacro_path]),
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }]
        ),

        Node(
            package='dsw_lio_sam',
            executable='dsw_lio_sam_imuPreintegration',
            name='dsw_lio_sam_imuPreintegration',
            parameters=[parameter_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen'
        ),

        Node(
            package='dsw_lio_sam',
            executable='dsw_lio_sam_imageProjection',
            name='dsw_lio_sam_imageProjection',
            parameters=[parameter_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen'
        ),

        Node(
            package='dsw_lio_sam',
            executable='dsw_lio_sam_featureExtraction',
            name='dsw_lio_sam_featureExtraction',
            parameters=[parameter_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen'
        ),

        Node(
            package='dsw_lio_sam',
            executable='dsw_lio_sam_mapOptimization',
            name='dsw_lio_sam_mapOptimization',
            parameters=[parameter_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen'
        ),

        Node(
            package='dsw_lio_sam',
            executable='dsw_lio_sam_semanticBridge',
            name='pseudo_semantic_bridge',
            parameters=[parameter_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen',
            condition=IfCondition(PythonExpression(
                ["'", LaunchConfiguration('start_pseudo_semantic_bridge'), "' == 'true'"]))
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_rviz'))
        )

    ])
