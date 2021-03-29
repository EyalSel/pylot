from setuptools import find_packages, setup

setup(
    name="erdos-pylot",
    version="0.3.1",
    author="Pylot Team",
    description=("A platform for developing autonomous vehicles."),
    long_description=open("README.md").read(),
    url="https://github.com/erdos-project/pylot",
    keywords=("autonomous vehicles driving python CARLA simulation"),
    packages=find_packages(),
    license="Apache 2.0",
    install_requires=[
        "absl-py",
        "cvxpy",
        "erdos>=0.3.1",
        "gdown",
        "lapsolver",
        "motmetrics",
        "numpy<1.17",  # Update to newer numpy version once we switch to tf2
        "open3d-python==0.5.0.0",
        "opencv-python>=4.1.0.25",
        "opencv-contrib-python>=4.1.0.25",
        "pillow>=6.2.2",
        "pycocotools",
        "pygame==1.9.6",
        "pytest",
        "scikit-image<0.15",
        "scipy==1.2.2",
        "shapely==1.6.4",
        "tensorflow-gpu==1.15.4",
        "torch==1.4.0",
        "torchvision==0.5.0",
        ##### Tracking dependencies #####
        "Cython",
        "filterpy==1.4.1",
        "imgaug==0.2.8",
        "matplotlib==2.2.4",
        "nonechucks==0.3.1",
        "nuscenes-devkit"
        "progress",
        "pyquaternion",
        "scikit-learn==0.22.2",
        ##### CARLA dependencies #####
        "networkx==2.2",
    ],
)
