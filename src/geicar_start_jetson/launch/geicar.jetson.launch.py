from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
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

    # LiDAR
    sllidar_node = Node(
        package="sllidar_ros2",
        executable="sllidar_node",
        parameters=[{
            "channel_type": "serial",
            "serial_port": "/dev/ttyUSB0",
            "serial_baudrate": 256000,
            "frame_id": "scan",
            "inverted": False,
            "angle_compensate": True,
        }],
        emulate_tty=True
    )

    # TF statique
    static_tf_scan = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_scan",
        arguments=[
            "--x", "-0.5",
            "--y", "0",
            "--z", "0.5",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "base_link",
            "--child-frame-id", "scan",
        ],
        emulate_tty=True
    )

    # rf2o_laser_odometry
    rf2o_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("rf2o_laser_odometry"),
                "launch",
                "rf2o_laser_odometry.launch.py"
            )
        )
    )

    return LaunchDescription([
        usb_cam_node_exe,
        audio_capture_node,
        sllidar_node,
        static_tf_scan,
        rf2o_launch,
    ])
