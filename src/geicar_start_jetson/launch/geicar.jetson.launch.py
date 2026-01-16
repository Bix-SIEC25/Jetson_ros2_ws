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
            "image_width": 640,
            "image_height": 480,
            "framerate": 15.0, # 5 10 15 20 30
            "pixel_format": "mjpeg2rgb"
        }],
        emulate_tty=True
    )


    # Micro
    audio_capture_node = Node(
        package="audio_common",
        executable="audio_capturer_node",
        parameters=[{
            "device": -1
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

    cmd_vel_to_motors = Node(
        package="cmd_vel_conv",
        executable="cmd_vel_to_motors_node",
        parameters=[{
            "max_linear_speed":  0.6,  # à adapter
            "max_angular_speed": 1.0,  # à adapter
        }],
        emulate_tty=True
    )
    
    watchdog = Node(
        package="watchdog",
        executable="watchdog",
        emulate_tty=True
    )

    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "slam": "False",
            "map": "/home/jetson/ros2_ws/src/slam/gei_0.yaml",
            "params_file": "/home/jetson/ros2_ws/src/slam/nav2.yaml",
        }.items(),
    )

    # --------- MACHINE A ETAT DE L'IA ---------

    state_machine = Node(
        package="ai_pkg",
        executable="ai_scenario",
        emulate_tty=True
    )
    

    return LaunchDescription([
        usb_cam_node_exe,
        # audio_capture_node,
        sllidar_node,
        static_tf_scan,
        rf2o_launch,
        cmd_vel_to_motors,
        nav2_launch,
        # watchdog,
        state_machine,
    ])
