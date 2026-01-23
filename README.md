# Jetson_ros2_ws

Workspace **ROS2 Humble** destiné à faire tourner la stack "robot autonome" sur **NVIDIA Jetson** (capteurs + navigation + IA), avec notamment :

- **LiDAR** (publication `/scan`)
- **Caméra USB** (publication `/image_raw`)
- **Odométrie LiDAR** via **rf2o** (Laser Odometry)
- **Navigation** via **Nav2** (AMCL + planner + controller + costmaps)
- **Conversion `cmd_vel` → commande moteurs** (publication `/motors_order_raw`)
- **Capture audio** directement intégrée dans le ai_pkg

> Objectif : une seule commande de launch pour démarrer la pile complète sur la Jetson.

## Arborescence logique (ce que contient le workspace)

Les fichiers fournis dans ce workspace montrent au minimum ces briques :

- `car_description/`  
  - `urdf/car.urdf.xacro` : description TF minimale (base + LiDAR)
  - `launch/robot_state_publisher.launch.py` : publication `/tf` via URDF

- `cmd_vel_conv/`  
  - `src/cmd_vel_to_motors_node.cpp` : conversion `geometry_msgs/Twist` → `interfaces/MotorsOrder`

- `slam/`
  - `nav2.yaml` : configuration Nav2 (AMCL, planner, controller, costmaps…)
  - `gei_0.yaml` : carte (map server)

- `geicar_bringup/` 
  - `launch/geicar.jetson.launch.py` : launch "full bringup" Jetson


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
ros2 launch geicar_start geicar.jetson.launch.py
```

---

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


## Topics, messages et TF

### Commande robot
- **Entrée standard** : `/cmd_vel` (`geometry_msgs/Twist`)
- **Sortie commande bas niveau** : `/motors_order_raw` (`interfaces/msg/MotorsOrder`)

### Conversion `cmd_vel` → PWM (node `cmd_vel_to_motors_node`)
Paramètres (par défaut) :
- `max_throttle = 1.0` (m/s)
- `max_steering = 0.5` (rad/s)
- `steering_gain = 1.4`
- throttle : `center=50`, `range=50`  → PWM dans **[0..100]** avec stop à 50
- steering : `center=0`, `range=100` → PWM dans **[-100..100]** avec neutre à 0

### Frames TF
- `base_link` : base du robot
- `scan` : frame du LiDAR (utilisée dans le launch et dans `sllidar_node`)
- Le TF statique actuel publie `base_link -> scan`.

✅ Recommandation : **choisir un seul frame LiDAR** (`scan` *ou* `laser`) et être cohérent partout :
- soit tu modifies le xacro pour publier `base_link -> scan`
- soit tu gardes `laser` dans l’URDF et tu adaptes `frame_id` côté LiDAR + TF statique

---

## Nav2 (AMCL + planner + controller)

Le fichier `nav2.yaml` contient les paramètres principaux :
- `amcl` (localisation)
- `planner_server` (planner)
- `controller_server` (suivi de trajectoire)
- `global_costmap` / `local_costmap`
- `bt_navigator` / `behavior_server`

### À vérifier / adapter
- **Chemins map & params** : dans `geicar.jetson.launch.py`, les chemins sont **absolus**.  
  Si tu déplaces le workspace ou changes d’utilisateur, ça cassera.

👉 Bonne pratique (à faire quand tu auras le temps) : remplacer par des chemins basés sur `FindPackageShare(...)` pour rendre le launch portable.

---
