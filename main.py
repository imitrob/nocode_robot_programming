#!/usr/bin/env python3
'''Recording trajectories and storing them into a databaseself.

conda activate gesturenlu2
cd ~/lfd_ws/
source install/setup.bash
ros2 launch object_localization box_localization_launch.py

conda activate gesturenlu2
cd ~/lfd_ws/
source install/setup.bash
python /home/imitlearn/lfd_ws/src/nocode_robot_programming/main.py
'''

# 1. Record with all modalities:

import rclpy
from skills_manager.lfd import LfD
from skills_manager.risk_aware_lfd.ralfd import RALfD
from joystick import JoystickControl
from gestures import TeleoperationByDrawing

from gesture_detector.hand_processing.hand_listener import HandListener
from skills_manager.ros_param_manager import set_remote_parameters

class MultimodalLfD(HandListener, RALfD):
    def __init__(self):
        super(MultimodalLfD, self).__init__()

        # input modalities has `step` function that:
        # 1. reads its inputs, 2. directly controls the robot
        self.gestures = TeleoperationByDrawing(self)
        self.joystick = JoystickControl(self)
        self.kinesthetics = None

        self.input_modality = None

    def set_to_gestures(self):
        self.input_modality = self.gestures

    def set_to_joystick(self):
        self.input_modality = self.joystick

    def set_to_kinesthetics(self):
        self.input_modality = self.kinesthetics 


def main():
    rclpy.init()
    try:
        lfd = MultimodalLfD()
        lfd.start()
        set_remote_parameters(lfd, ["position_x", "position_y", "position_z", "orientation_x", "orientation_y", "orientation_z", "orientation_w"],
        [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], server="localizer_node")
        lfd.move_template_start()
        lfd.set_to_joystick()

        name_skill = lfd.declare_parameter_and_get('name_skill', "peg_pick")
        print(f"Recording skill: {name_skill}", flush=True)

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