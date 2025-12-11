from setuptools import find_packages, setup
import os

package_name = 'ai_pkg'

# --- data_files de base ROS 2 ---
data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

# --- ajout récursif du modèle Vosk ---
# arborescence attendue :
# ai_pkg/
#   models/
#     vosk-model-small-en-us-0.15/
#       am/
#       conf/
#       graph/
#       ivector/
#       README
model_root = os.path.join('models', 'vosk-model-small-en-us-0.15')

if os.path.isdir(model_root):
    for root, dirs, files in os.walk(model_root):
        if not files:
            continue  # on ignore les dossiers vides
        # root ex: "models/vosk-model-small-en-us-0.15/am"
        install_dir = os.path.join('share', package_name, root)
        src_files = [os.path.join(root, f) for f in files]
        data_files.append((install_dir, src_files))


# --- install embeddings & names .npy files ---
embeddings_file = os.path.join('models', 'embeddings.npy')
names_file = os.path.join('models', 'names.npy')

if os.path.isfile(embeddings_file) and os.path.isfile(names_file):
    data_files.append(
        (os.path.join('share', package_name, 'models'),
         [embeddings_file, names_file])
    )


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=[
        'setuptools',
        # ajoute ici si besoin:
        # 'rclpy',
        'vosk',
        'audio_common_msgs',
    ],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='187615643+LahnM@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vocal_recognition = ai_pkg.vocal_recognition:main',
            'face_recognition_node = ai_pkg.face_recognition_node:main',
            'fall_detection = ai_pkg.fall_detection:main',
            'Dialogue_recognition = ai_pkg.Dialogue_recognition:main',
            'ai_scenario = ai_pkg.main_scenario:main',
        ],
    },
)
