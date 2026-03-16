# Nocode Robot Programming

Conditional Incremental Programming with vision-based branching.

You will need Franka Emika Panda Robot and robot-mounted RealSense Camera (you need to change device serial ID at `camera_launch.py`). Optionally, we use Logitech joystick (you need to change device path [here](nocode_robot_programming/joystick.py)) and optionally Leap Motion for hand gestures.

Teaching probing task: Task-graph with three skill variants:
<img src="./probing.gif" alt="Teaching_probing_task" />

## Installation:

```
mkdir -p lfd_ws/src
cd lfd_ws/src
git clone https://github.com/imitrob/franka_learning_from_demonstration_ros2 -b program
git clone https://github.com/imitrob/nocode_robot_programming.git

conda env create -f nocode_robot_programming/environment.yml
# Is quicker with: mamba env create -f nocode_robot_programming/environment.yml
conda activate nocodeprogram
cd ..
colcon build --symlink-install
source install/setup.bash
```
ROS2 installs the packages to build folder. Make a symbolic links to use materials such as trajectories, configs, templates.
```
ln -s ~/lfd_ws/src/franka_learning_from_demonstrations/trajectory_data/trajectories ~/lfd_ws/build/trajectory_data/trajectories
```
Please remember to source the workspace in every terminal you open.

## Notebooks

1. [Preparation notebook](tests/00_robot_check.ipynb): See the robot & gripper moving
2. [Teaching and Execution](tests/01_user_study.ipynb): User-study dashboard
3. [Dataset check](tests/02_dataset_check.ipynb): See collected dataset as train/test sets
4. [Evaluation](tests/03_dataset_eval.ipynb) of collected data

## Datasets 

Download our user-study dataset: `https://drive.google.com/file/d/1ij-c3Ahr6vsKfk7Wq1fUlYrME872Pbxl/view?usp=sharing`, extract, and copy it to `franka_learning_from_demonstration_ros2/trajectory_data/trajectories` folder.

## Notes:

- In teaching, state decider is trained once before task execution. 

## TODOs:

- [ ] This repo uses `program` branch `franka_learning_from_demonstration_ros2`: I should do pull request.
