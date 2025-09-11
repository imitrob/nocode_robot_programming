
import argparse

from copy import deepcopy
import time
import numpy as np
from spatialmath import UnitQuaternion
import spatialmath as sm 

import rclpy
import threading

from playsound import playsound

import numpy as np

def transform_leap_to_scene(data, scale=1.0, start=[0.5, 0.0, 0.2]):
    x, y, z = data[0], data[1], data[2]
    
    x_ =  x/1000
    y_ = -z/1000
    z_ =  y/1000

    x__ = np.dot([x_,y_,z_], [0,-1, 0])*scale + start[0]
    y__ = np.dot([x_,y_,z_], [1, 0, 0])*scale + start[1]
    z__ = np.dot([x_,y_,z_], [0, 0, 1])*scale + start[2]

    data[0] = x__
    data[1] = y__
    data[2] = z__

    return data

class Servo():
    def __init__(self,
                 node,
                 teleop_hand: str = "l", 
                 aux_hand: str = "r",
                 link_gesture: str = "grab_strength",
                 teleop_scale: float = 1.0,
                 teleop_rotate_eef: bool = True,
                 ):
        """
        Panda:
            self.move_to_pose(position, orientation) # position (float[3]), orientation (float[4])
            self.grasp(width, speed, force, epsilon_inner, epsilon_outer)

        Args:
            teleop_hand (str, optional): Hand used to teleoperate. 
                Defaults to "l" left hand. "r" for right hand. "" to disable teleop.
            aux_hand (str, optional): Hand used for auxiliary action (gripper open/close).
                Defaults to "l" left hand. "r" for right hand. "" to disable aux action.
            link_gesture (str, optional): Gesture to trigger teleoperation
                Defaults to "grab_strength" - Grab gesture triggers teleoperation.
            teleop_rotate_eef (bool, optional): Reads angle of hand and rotates 7th joint.
                Defaults to True.
        """
        super(Servo, self).__init__()

        self.node = node

        self.teleop_hand = teleop_hand 
        self.aux_hand = aux_hand
        
        self.link_gesture = link_gesture
        self.teleop_scale = teleop_scale
        self.teleop_rotate_eef = teleop_rotate_eef
        
        self.scene_anchor_save = [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0] # x,y,z,qx,qy,qz,qw [m] wrt. robot base
        self.tgoal_pose = [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        self.eef_rot = 0.0
        self.teleop_trigger = False

        play_thread = threading.Thread(target=self.playontrigger, args=(), daemon=True)
        play_thread.start()


    def is_hand_visible(self, hand):
        if (self.node.hand_frames and 
            self.node.hand_frames[-1] and
            getattr(self.node.hand_frames[-1],hand) and
            getattr(self.node.hand_frames[-1],hand).visible):
            return True
        return False

    def is_gesture_activated(self, hand, gesture):
        if self.is_hand_visible(hand):
            if getattr(getattr(self.node.hand_frames[-1],hand),gesture) > 0.8:
                return True
        return False

    def playontrigger(self):
        while True:
            if self.teleop_trigger and self.is_hand_visible(self.teleop_hand):
                playsound('/usr/share/sounds/Yaru/stereo/bell.oga', block=True)
            else:
                time.sleep(0.5)

    def teleoperation_step(self, trigger):
        goal_pose = transform_leap_to_scene(
            getattr(self.node.hand_frames[-1],self.teleop_hand).palm_position(),
            self.teleop_scale)
        
        self.node.move_to_pose(position=goal_pose, orientation=[1.0, 0.0, 0.0, 0.0], speed_factor=0.05)

    def step(self):
        if self.is_hand_visible(self.teleop_hand):
            trigger = self.is_gesture_activated(self.teleop_hand, self.link_gesture)
            self.teleoperation_step(trigger)
        else:
            self.is_drawing = False
        
        if self.is_hand_visible(self.aux_hand):
            grab_strength = getattr(self.node.hand_frames[-1], self.aux_hand).grab_strength

            # OPTION: Close gripper proportionally with grab strength:
            # self.gripper.grasp(width=(1.-grab_strength)/1, speed=0.2, force=10, epsilon_inner=0.04, epsilon_outer=0.04)
            if grab_strength > 0.8:
                if not self.node.gripper.read_once().is_grasped:
                    self.node.gripper.grasp(width=0, speed=0.2, force=10, epsilon_inner=0.04, epsilon_outer=0.04)
            elif grab_strength < 0.2:
                self.node.gripper.move(0.08, 0.2)


class TeleoperationByDrawing(Servo):
    """Live mode is enabled only, when link_gesture is activated.
    """    
    def teleoperation_step(self, trigger):
        self.teleop_trigger = trigger
        if trigger:
            
            mouse3d = transform_leap_to_scene(
                getattr(self.node.hand_frames[-1],self.teleop_hand).palm_pose_list(),
                self.teleop_scale
            )

            if self.teleop_rotate_eef:
                x,y = self.node.hand_frames[-1].r.direction()[0:2]
                angle = np.arctan2(y,x)

            if not self.is_drawing: # init anchor
                self.anchor = mouse3d
                self.scene_anchor = deepcopy(self.scene_anchor_save)
                self.is_drawing = True

                if self.teleop_rotate_eef:
                    self.eef_rot_scene = deepcopy(self.eef_rot)
                    self.live_mode_drawing_eef_rot_anchor = angle

            #goal_pose = goal_pose + (mouse3d - self.anchor)
            goal_pose = deepcopy(self.scene_anchor)
            goal_pose[0] += (mouse3d[0] - self.anchor[0])
            goal_pose[1] += (mouse3d[1] - self.anchor[1])
            goal_pose[2] += (mouse3d[2] - self.anchor[2])

            if self.teleop_rotate_eef:
                self.eef_rot = deepcopy(self.eef_rot_scene)
                self.eef_rot += (angle - self.live_mode_drawing_eef_rot_anchor)

            q = UnitQuaternion([0.0,0.0,1.0,0.0])
            rot = sm.SO3(q.R) * sm.SO3.Rz(self.eef_rot)
            goal_pose[3],goal_pose[4],goal_pose[5],goal_pose[6] = UnitQuaternion(rot).vec_xyzs
            
            # Save cage
            goal_pose = np.clip(
                goal_pose,
                #        [x  , y   , z   , no limits on rotation]
                np.array([0.2, -0.4, 0.03, -10, -10, -10, -10]),
                np.array([0.6,  0.4, 0.4,   10,  10,  10,  10])
            )
            self.tgoal_pose = goal_pose
        else:
            self.scene_anchor_save = self.tgoal_pose
            self.is_drawing = False

        self.node.move_to_pose(position=(
            self.tgoal_pose[0], 
            self.tgoal_pose[1], 
            self.tgoal_pose[2]), 
            orientation=[1.0, 0.0, 0.0, 0.0], 
            speed_factor=0.01
        )



def main(args):
    rclpy.init()

    teleop = TeleoperationByDrawing(
        teleop_hand = args['teleop_hand'], 
        aux_hand = args['aux_hand'],
        link_gesture = args['link_gesture'],
        teleop_scale = args['teleop_scale'],
        teleop_rotate_eef = args['teleop_rotate_eef'],
    )

    while True:
        teleop.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run teleoperation node",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--teleop_hand",
        default="r",
        choices=["l", "r", ""],
    )
    parser.add_argument(
        "--aux_hand",
        default="l",
        choices=["l", "r", ""],
        help="Hand that opens and closes the gripper."
    )
    parser.add_argument(
        "--link_gesture",
        default="grab_strength",
        choices=["grab_strength", # hand closed
                 "pinch_strength", # thumb and point fingers touching 
                ],
        help="The gesture activates teleoperation."
    )
    parser.add_argument(
        "--teleop_scale",
        default=1.0,
        help="Stretching hand move distance to task space distance."
    )
    parser.add_argument(
        "--teleop_rotate_eef",
        default=True,
    )

    main(vars(parser.parse_args()))
