# Nocode Robot Programming

Conditional Incremental Programming with vision-based branching.

You will need Franka Emika Panda Robot and robot-mounted RealSense Camera (you need to change device serial ID at `camera_launch.py`). Optionally, we use Logitech joystick (you need to change device path [here](nocode_robot_programming/joystick.py)) and optionally Leap Motion for hand gestures.

Teaching probing task: Task-graph with three skill variants:
<img src="./probing.gif" alt="Teaching_probing_task" />

## Installation:

```
mkdir -p lfd_ws/src
cd lfd_ws/src
git clone https://github.com/imitrob/franka_learning_from_demonstrations_ros2 -b program
git clone https://github.com/imitrob/nocode_robot_programming.git

conda env create -f nocode_robot_programming/environment.yml
# Is quicker with: mamba env create -f nocode_robot_programming/environment.yml
conda activate nocodeprogram
cd ..
colcon build --symlink-install --cmake-args -DPython3_FIND_VIRTUALENV=ONLY
source install/setup.bash
```
ROS2 installs the packages to build folder. Make a symbolic links to use materials such as trajectories, configs, templates.
```
ln -s ~/lfd_ws/src/franka_learning_from_demonstrations_ros2/trajectory_data/trajectories ~/lfd_ws/build/trajectory_data/trajectories
```
Please remember to source the workspace in every terminal you open. A handy alias for `~/.bashrc`:

```shell
alias lfd='conda deactivate; conda activate gesturenlu2; source ~/lfd_ws/install/setup.bash; [ -f /etc/udev/rules.d/99-realsense-no-suspend.rules ] || { echo '\''ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="8086", TEST=="power/control", ATTR{power/control}="on"'\'' | sudo tee /etc/udev/rules.d/99-realsense-no-suspend.rules && sudo udevadm control --reload && sudo udevadm trigger; }'
```
- Disables RealSense autosuspend, please check idVendor is correct. 

## Notebooks

Launch VSCode while sourcing ROS env first: `source install/setup.bash; code src/`

1. [Preparation notebook](tests/00_robot_check.ipynb): See the robot & gripper moving
2. [Teaching and Execution](tests/01_user_study.ipynb): User-study dashboard
3. [Dataset check](tests/02_dataset_check.ipynb): See collected dataset as train/test sets
4. [Evaluation](tests/03_evaluation.ipynb) of collected data

## Datasets

Download our user-study [dataset](`https://drive.google.com/file/d/1ij-c3Ahr6vsKfk7Wq1fUlYrME872Pbxl/view?usp=sharing`) extracted to [/trajectories](`franka_learning_from_demonstration_ros2/trajectory_data/trajectories`) folder. Use the following commands to do that:

```shell
dst="$HOME/lfd_ws/src/franka_learning_from_demonstrations_ros2/trajectory_data/trajectories"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$dst"
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1ij-c3Ahr6vsKfk7Wq1fUlYrME872Pbxl&export=download&confirm=t" -O "$tmp/dataset.zip"
unzip -q "$tmp/dataset.zip" -d "$tmp/extracted"
cp -r "$tmp/extracted"/. "$dst"/
```

`tests/03_dataset_eval.ipynb` uses `trajectory_criteria.csv` in the trajectories
folder. The notebook creates/syncs this file with columns
`filename,use,criterion,discard_reason`; set `use` to `0` for a corrupted
trajectory.



