from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()


    system_check_ack_node = Node(
        package="system_check_ack",
        executable="system_check_ack_node",
        emulate_tty=True
    )

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


    sllidar_node = Node(
        package="sllidar_ros2",
        executable="sllidar_node",
        parameters=[{
            "channel_type": "serial",
            "serial_port": "/dev/ttyUSB0",
            "serial_baudrate": 256000,
            "frame_id": "scan",
            "inverted": False,
            "angle_compensate": True
        }],
        emulate_tty=True
    )

    wheel_odom_node = Node(
        package="wheel_odom",
        executable="wheel_odom_node",
        parameters=[{
            "wheel_radius": 0.095,        # à ajuster
            "wheel_separation": 0.50,    # à ajuster
            "ticks_per_rev": 36,
            "frame_id": "odom",
            "child_frame_id": "base_link",
        }],
        emulate_tty=True
    )

    laser_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_laser_tf",
        arguments=[
            "-0.5", "0.0", "0.5",   # x y z
            "0", "0", "0",          # roll pitch yaw
            "base_link",            # parent
            "scan"                  # enfant
        ]
    )


    odom_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="odom_to_base_link_tf",
        arguments=[
            "0.0", "0.0", "0.0",    # x y z
            "0.0", "0.0", "0.0",    # roll pitch yaw
            "odom",                 # parent
            "base_link"             # child
        ]
    )

    odom_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="odom_to_base_link_tf",
        arguments=["0","0","0","0","0","0","odom","base_link"]
    )


    ld.add_action(system_check_ack_node)
    ld.add_action(usb_cam_node_exe)
    ld.add_action(audio_capture_node)
    ld.add_action(sllidar_node)
    # ld.add_action(wheel_odom_node)
    # ld.add_action(laser_static_tf)
    # ld.add_action(odom_static_tf)
    ld.add_action(odom_static_tf)

    return ld