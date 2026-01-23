# Jetson_ros2_ws

Workspace **ROS2 Humble** destiné à faire tourner la stack "robot autonome" sur **NVIDIA Jetson Orin Nano** (capteurs + navigation + IA), avec notamment :

- **LiDAR** (publication `/scan`)
- **Caméra USB** (publication `/image_raw`)
- **Odométrie LiDAR** via **rf2o** (Laser Odometry)
- **Navigation** via **Nav2** (AMCL + planner + controller + costmaps)
- **Conversion `cmd_vel` → commande moteurs** (publication `/motors_order_raw`)
- **Capture audio** directement intégrée dans le ai_pkg

> Objectif : une seule commande de launch pour démarrer la pile complète sur la Jetson.

## Arborescence

- `src/` : paquets ROS 2
  - `ai_pkg/` : Un des pkgs les plus important du ws, il contient toute la logique de la machine à état
  - `audio_common/` : envoie des messages TTS et des sons de klaxons à l'enceinte
  - `car_description/` : description de la size de la voiture
  - `interfaces/` : messages ROS personnalisés (`.msg`)
  - `system_check/` : check communications + report
  - `watchdog/` : surveillance (selon implémentation)
  - `rf2o_laser_odometry/` : permet de publier l'odométrie de la voiture
  - `geicar_start_jetson/` : launch principal
  - `sllidar_ros2/` : permet de publier le topic `\scan` contenant les frames du LiDAR


## Quickstart

### 1) Dépendances
- ROS 2 installé et sourcé (`/opt/ros/$ROS_DISTRO`)
- Outils build : `colcon`, `rosdep`
- Python deps :
  - `requests`
  - `webrtcvad`

### 2) Build
```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 3) Lancer la stack complète Jetson
Le launch principal démarre caméra + LiDAR + TF + rf2o + Nav2 + scénario IA :
```bash
ros2 launch geicar_start_jetson geicar.jetson.launch.py
```


## Launch

### `geicar.jetson.launch.py`
Démarre :

1. **Caméra** (package `usb_cam`)
   - node : `usb_cam_node_exe`
   - params :
     - `video_device: /dev/video0`
     - `pixel_format: uyvy`
     - `image_width: 1920`
     - `image_height: 1080`

2. **LiDAR** (package `sllidar_ros2`)
   - node : `sllidar_node`
   - params :
     - `serial_port: /dev/ttyUSB0`
     - `serial_baudrate: 256000`
     - `frame_id: scan`
     - `inverted: false`
     - `angle_compensate: true`

3. **TF statique** base → scan
   - `tf2_ros/static_transform_publisher`
   - valeurs actuelles : `x=0.5, y=0, z=0.2, roll=pitch=yaw=0`
   - frames : `base_link` → `scan`

4. **Odométrie LiDAR** (rf2o)
   - inclut `rf2o_laser_odometry/launch/rf2o_laser_odometry.launch.py`

5. **Conversion commande** `cmd_vel` → moteurs (package `cmd_vel_conv`)
   - node : `cmd_vel_to_motors_node`
   - pub : `motors_order_raw`

6. **Nav2 bringup**
   - `nav2_bringup/bringup_launch.py`
   - paramètres :
     - `map: /home/jetson/ros2_ws/src/slam/gei_0.yaml`
     - `params_file: /home/jetson/ros2_ws/src/slam/nav2.yaml`

7. **IA / scénario**
   - package `ai_pkg`, executable `ai_scenario`

