#!/usr/bin/env python3
'''Recording trajectories and storing them into a databaseself.

sudo leapd

conda activate gesturenlu2
cd ~/lfd_ws/
source install/setup.bash
ros2 run gesture_detector leap

conda activate gesturenlu2
cd ~/lfd_ws/
source install/setup.bash
ros2 launch object_localization box_localization_launch.py

conda activate gesturenlu2
cd ~/lfd_ws/
source install/setup.bash
python /home/imitlearn/lfd_ws/src/nocode_robot_programming/nocode_robot_programming/main.py
'''
import rclpy
from skills_manager.risk_aware_lfd.ralfd import RALfD
from skills_manager.ros_param_manager import set_remote_parameters

def main():
    rclpy.init()
    try:
        lfd = RALfD()
        lfd.start()
        set_remote_parameters(lfd, ["position_x", "position_y", "position_z", "orientation_x", "orientation_y", "orientation_z", "orientation_w"],
        [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], server="localizer_node")
        lfd.move_template_start()
        
        name_skill = lfd.declare_parameter_and_get('name_skill', "peg_pick")
        print(f"Recording skill: {name_skill}", flush=True)

        if not lfd.skill_exists():
            lfd.traj_rec()
            lfd.save(name_skill)
            input("ENTER to replay")

        lfd.move_template_start()
        success = lfd.play_skill(name_skill, None, localize_box=False)


    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    print("finished", flush=True)

if __name__ == '__main__':
    main()