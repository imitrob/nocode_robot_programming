
import argparse

from copy import deepcopy
import time
import numpy as np
from spatialmath import UnitQuaternion
import spatialmath as sm 

import rclpy
import threading

import numpy as np
from gesture_detector.hand_processing.hand_listener import HandListener
from nocode_robot_programming.feedback_sound import sound_thread

FREQ = 20 # Hz

def _unit(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / n if n > eps else v*0.0

def classify_hand_direction_asym(
    palm_vec, wrist_vec,
    sensor_up=np.array([0., 1., 0.]),
    sensor_right=np.array([1., 0., 0.]),
    bend_thresh_deg=7.0,           # default for up/down/left
    bend_thresh_right_deg=3.5,     # easier bend threshold for RIGHT
    strength_thresh=0.35,          # default strength
    strength_thresh_right=0.25     # easier strength for RIGHT (tune)
):
    f = _unit(wrist_vec)
    p = _unit(palm_vec)

    # Bend angle between forearm and palm (0 = straight)
    cross_fp = np.cross(f, p)
    dot_fp = float(np.dot(f, p))
    bend_rad = np.arctan2(np.linalg.norm(cross_fp), np.clip(dot_fp, -1.0, 1.0))
    bend_deg = np.degrees(bend_rad)

    # Lateral component (remove along-forearm)
    l = p - np.dot(p, f) * f
    l = _unit(l)
    if np.allclose(l, 0.0):
        return "straight", bend_deg, (0.0, 0.0)

    # Project sensor up/right into plane ⟂ f
    u_hat = _unit(sensor_up   - np.dot(sensor_up, f)   * f)
    r_hat = _unit(sensor_right- np.dot(sensor_right, f)* f)
    if np.allclose(u_hat, 0.0) or np.allclose(r_hat, 0.0):
        # rebuild a local basis if needed
        tmp = np.array([1.,0.,0.]) if abs(f[0]) < 0.9 else np.array([0.,1.,0.])
        r_hat = _unit(np.cross(f, tmp))
        u_hat = _unit(np.cross(r_hat, f))

    u_comp = float(np.dot(l, u_hat))   # +up,  -down
    r_comp = float(np.dot(l, r_hat))   # +right,-left

    candidates = {
        "up":    u_comp,
        "down": -u_comp,
        "right": r_comp,
        "left": -r_comp,
    }
    label = max(candidates, key=candidates.get)
    val   = candidates[label]

    # Per-label thresholds
    bend_req = {
        "up": bend_thresh_deg,
        "down": bend_thresh_deg,
        "left": bend_thresh_deg,
        "right": bend_thresh_right_deg,
    }[label]
    strength_req = {
        "up": strength_thresh,
        "down": strength_thresh,
        "left": strength_thresh,
        "right": strength_thresh_right,
    }[label]

    if bend_deg >= bend_req and val >= strength_req:
        return label, bend_deg, (u_comp, r_comp)
    else:
        return "straight", bend_deg, (u_comp, r_comp)



class TeleoperationByDrawing(HandListener):
    def __init__(self,
                 teleop_hand: str = "l", 
                 teleop_aux_hand: str = "r",
                 link_gesture: str = "grab_strength",
                 teleop_scale: float = 1.0,
                 teleop_rotate_eef: bool = True,
                 ):
        """
        Args:
            teleop_hand (str, optional): Hand used to teleoperate. 
                Defaults to "l" left hand. "r" for right hand. "" to disable teleop.
            teleop_aux_hand (str, optional): Hand used for auxiliary action (gripper open/close).
                Defaults to "l" left hand. "r" for right hand. "" to disable aux action.
            link_gesture (str, optional): Gesture to trigger teleoperation
                Defaults to "grab_strength" - Grab gesture triggers teleoperation.
            teleop_rotate_eef (bool, optional): Reads angle of hand and rotates 7th joint.
                Defaults to True.
        """
        super(TeleoperationByDrawing, self).__init__()
        self.teleop_hand = teleop_hand 
        self.teleop_aux_hand = teleop_aux_hand
        
        self.link_gesture = link_gesture
        self.teleop_scale = teleop_scale
        self.teleop_rotate_eef = teleop_rotate_eef
        
        self.scene_anchor_save = None #[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0] # x,y,z,qx,qy,qz,qw [m] wrt. robot base
        self.eef_rot = 0.0
        self.teleop_trigger = False

        play_thread = threading.Thread(target=sound_thread, args=(self,), daemon=True)
        play_thread.start()

    def teleop_has_control(self):
        return (self.teleop_trigger and self.is_hand_visible(self.teleop_hand))
            

    def is_hand_visible(self, hand):
        if (self.hand_frames and 
            self.hand_frames[-1] and
            getattr(self.hand_frames[-1],hand) and
            getattr(self.hand_frames[-1],hand).visible):
            return True
        return False

    def is_gesture_activated(self, hand, gesture):
        if self.is_hand_visible(hand):
            if getattr(getattr(self.hand_frames[-1],hand),gesture) > 0.8:
                return True
        return False

    def teleop_start(self):
        self.teleop_thr = threading.Thread(target=self.teleop_thread_start, daemon=True)
        self.teleop_thr.start()

    def teleop_stop(self):
        self.teleop_thr.join(timeout=1)

    def teleop_thread_start(self):
        self.scene_anchor_save = [*self.panda.get_position(), *self.panda.get_orientation(scalar_first=False)]

        try:
            while True:
                self.teleop_step()
                time.sleep(1./FREQ)
        except KeyboardInterrupt:
            pass

    def teleop_step(self):
        if self.is_hand_visible(self.teleop_hand):
            self.pause = False
            grab_strength = getattr(self.hand_frames[-1], self.teleop_hand).grab_strength
            
            trigger = self.is_gesture_activated(self.teleop_hand, self.link_gesture)
            self.teleop_position_compute(trigger)
        
        else:
            self.is_drawing = False
        
        if self.is_hand_visible(self.teleop_aux_hand):
            grab_strength = getattr(self.hand_frames[-1], self.teleop_aux_hand).grab_strength
            pinch_strength = getattr(self.hand_frames[-1], self.teleop_aux_hand).pinch_strength
            if grab_strength == 0.0:
                self.pause = True # stops the execution
            else:
                self.pause = False

            if grab_strength > 0.8 or pinch_strength > 0.8:
                if not self.gripper_state.is_grasped:
                    self.feedback_gripper = "grasp"
            elif grab_strength < 0.2 or pinch_strength < 0.2:
                self.feedback_gripper = "open"

    def teleop_position_compute(self, trigger):
        self.teleop_trigger = trigger
        if trigger:
            
            mouse3d_ = 0.001 * np.array(getattr(self.hand_frames[-1],self.teleop_hand).palm_pose_list()) 

            mouse3d = np.array([0.,0.,0.])
            mouse3d[0] = -mouse3d_[0]
            mouse3d[1] = mouse3d_[2]
            mouse3d[2] = mouse3d_[1]

            if self.teleop_rotate_eef:
                x,y = self.hand_frames[-1].r.direction()[0:2]
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

            # Save cage
            # goal_pose = np.clip(
            #     goal_pose,
            #     #        [x  , y   , z   , no limits on rotation]
            #     np.array([0.2, -0.4, 0.05, -10, -10, -10, -10]),
            #     np.array([0.6,  0.4, 0.5,   10,  10,  10,  10])
            # )

            current_position = self.panda.get_position() 
            factor = 0.5
            self.feedback[0] = (goal_pose[0] - current_position[0]) * factor
            self.feedback[1] = (goal_pose[1] - current_position[1]) * factor
            self.feedback[2] = (goal_pose[2] - current_position[2]) * factor
        else:
            pitch = self.hand_frames[-1].l.palm_normal.pitch()
            yaw = self.hand_frames[-1].l.direction.yaw()
            # print(f"{'left' if pitch < 1.0 else ''}{'right' if pitch > 2.0 else ''} {'up' if yaw < 1.0 else ''}{'down' if yaw > 2.0 else ''}")
            if pitch < 1.0:
                self.feedback[4] = -0.2
            elif pitch > 2.0:
                self.feedback[4] = 0.2
            elif yaw < 1.0:
                self.feedback[3] = -0.1
            elif yaw > 2.0:
                self.feedback[3] = 0.1
            else:
                self.feedback[3] = 0.0
                self.feedback[4] = 0.0

            self.scene_anchor_save = [*self.panda.get_position(), *self.panda.get_orientation(scalar_first=False)]  #self.feedback
            self.is_drawing = False

        if sum(np.absolute(self.feedback)) > 0:
            self.modality_in_control = 'gestures'


def main(args):
    rclpy.init()

    teleop = TeleoperationByDrawing(
        teleop_hand = args['teleop_hand'], 
        teleop_aux_hand = args['teleop_aux_hand'],
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
        "--teleop_aux_hand",
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
