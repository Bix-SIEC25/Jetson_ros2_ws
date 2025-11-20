from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # Caméra USB
    usb_cam_node_exe = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        parameters=[{
            "image_width": 1280,
            "image_height": 960,
            "pixel_format": "mjpeg2rgb"
        }],
        emulate_tty=True
    )

    # Micro
    audio_capture_node = Node(
        package="audio_common",
        executable="audio_capturer_node",
        parameters=[{
            "device": 0
        }],
        remappings=[
            ('/audio', '/audio_mic')
        ],
        emulate_tty=True
    )

    # 1) LIDAR
    sllidar_node = Node(
        package='sllidar_ros2',
        executable='sllidar_node',
        name='sllidar_node',
        output='screen',
        parameters=[
            {'channel_type': 'serial'},
            {'serial_port': '/dev/ttyUSB0'},
            {'serial_baudrate': 256000},
            {'frame_id': 'scan'},
            {'inverted': False},
            {'angle_compensate': True},
        ]
    )

    # 2) TF statique base_link -> scan
    static_tf_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_laser',
        arguments=[
            '--x', '-0.5',
            '--y', '0.0',
            '--z', '0.5',
            '--roll', '0',
            '--pitch', '0',
            '--yaw', '0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'scan'
        ]
    )

    # 3) TON node d’odom roues (remplace rf2o)
    wheel_odom_node = Node(
        package='wheel_odom',          # <-- ton package
        executable='wheel_odom_node',  # <-- ton exécutable
        name='wheel_odom_node',
        output='screen',
        parameters=[
            {'wheel_radius': 0.095},
            {'wheel_separation': 0.50},
            {'ticks_per_rev': 36},
            {'frame_id': 'odom'},
            {'child_frame_id': 'base_link'},
        ]
    )

    return LaunchDescription([
        usb_cam_node_exe,
        audio_capture_node,
        sllidar_node,
        static_tf_laser,
        wheel_odom_node,
    ])
