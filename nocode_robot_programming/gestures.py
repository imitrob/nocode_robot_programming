
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
                 teleop_scale: float = 0.3,
                 ):
        """
        Args:
            teleop_hand (str, optional): Hand used to teleoperate. 
                Defaults to "l" left hand. "r" for right hand. "" to disable teleop.
            teleop_aux_hand (str, optional): Hand used for auxiliary action (gripper open/close).
                Defaults to "l" left hand. "r" for right hand. "" to disable aux action.
            link_gesture (str, optional): Gesture to trigger teleoperation
                Defaults to "grab_strength" - Grab gesture triggers teleoperation.
        """
        super(TeleoperationByDrawing, self).__init__()
        self.teleop_hand = teleop_hand 
        self.teleop_aux_hand = teleop_aux_hand
        
        self.link_gesture = link_gesture
        self.teleop_scale = teleop_scale
        
        self.scene_anchor_save = None #[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0] # x,y,z,qx,qy,qz,qw [m] wrt. robot base
        self.teleop_trigger = False

        play_thread = threading.Thread(target=sound_thread, args=(self,), daemon=True)
        play_thread.start()

        self.teleop_thr_running = False

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
            if getattr(getattr(self.hand_frames[-1],hand),gesture) > 0.5:
                return True
        return False

    def teleop_start(self):
        if not self.teleop_thr_running:
            self.teleop_thr_running = True
            self.teleop_thr = threading.Thread(target=self.teleop_thread_start, daemon=True)
            self.teleop_thr.start()

    def teleop_stop(self):
        if self.teleop_thr_running:
            self.teleop_thr.join(timeout=1)
            self.teleop_thr_running = False

    def teleop_thread_start(self):
        self.scene_anchor_save = [*self.panda.get_position(), *self.panda.get_orientation(scalar_first=False)]

        try:
            while True:
                self.teleop_step()
                time.sleep(1./FREQ)
        except KeyboardInterrupt:
            pass

    def teleop_step(self):
        if self.is_hand_visible(self.teleop_hand) and not self.is_hand_visible(self.teleop_aux_hand):
            grab_strength = getattr(self.hand_frames[-1], self.teleop_hand).grab_strength
            
            trigger = self.is_gesture_activated(self.teleop_hand, self.link_gesture)
            self.teleop_position_compute(trigger)
        
        else:
            self.is_drawing = False
        
        if self.is_hand_visible(self.teleop_aux_hand) and self.is_hand_visible(self.teleop_hand):
            grab_strength = getattr(self.hand_frames[-1], self.teleop_aux_hand).grab_strength
            pinch_strength = getattr(self.hand_frames[-1], self.teleop_aux_hand).pinch_strength
            
            
            if len(self.hand_frames) >= 10:
                if self.hand_frames[-1].leapgestures.swipe.present:
                    self.pause = False

                yaw = getattr(self.hand_frames[-1], self.teleop_aux_hand).direction.yaw()
                this_frame_stop = getattr(self.hand_frames[-1], self.teleop_aux_hand).is_stop()
                prev_frame_stop = getattr(self.hand_frames[-10], self.teleop_aux_hand).is_stop()
                
                if grab_strength == 0.0 and this_frame_stop and prev_frame_stop and yaw < 1.0:
                    self.pause = True # stops the execution
                    self.end += 1

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

            if not self.is_drawing: # init anchor
                self.anchor = mouse3d
                self.scene_anchor = deepcopy(self.scene_anchor_save)
                self.is_drawing = True

            #goal_pose = goal_pose + (mouse3d - self.anchor)
            goal_pose = deepcopy(self.scene_anchor)

            self.gesture_feedback = [
                goal_pose[0] + (mouse3d[0] - self.anchor[0]) * 0.3,
                goal_pose[1] + (mouse3d[1] - self.anchor[1]) * 0.3,
                goal_pose[2] + (mouse3d[2] - self.anchor[2]) * 0.3,
            ]
            
        else:
            pitch = self.hand_frames[-1].l.palm_normal.pitch()
            yaw = self.hand_frames[-1].l.direction.yaw()
            # print(f"{'left' if pitch < 1.0 else ''}{'right' if pitch > 2.0 else ''} {'up' if yaw < 1.0 else ''}{'down' if yaw > 2.0 else ''}")
            if self.hand_frames[-1].leapgestures.circle.present and self.hand_frames[-1].leapgestures.circle.clockwise: # yaw < 1.0:
                if self.hand_frames[-1].leapgestures.circle.progress < 2:
                    feedback3 = -0.01
                    feedback4 = 0.0
                else:
                    feedback3 = -0.05
                    feedback4 = 0.0
            elif self.hand_frames[-1].leapgestures.circle.present and not self.hand_frames[-1].leapgestures.circle.clockwise: #yaw > 2.0:
                if self.hand_frames[-1].leapgestures.circle.progress < 2:
                    feedback3 = 0.01
                    feedback4 = 0.0
                else:
                    feedback3 = 0.05
                    feedback4 = 0.0
            elif pitch < 0.9:
                feedback3 = 0.0
                feedback4 = -0.05
            elif pitch > 2.1:
                feedback3 = 0.0
                feedback4 = 0.05
            else:
                feedback3 = 0.0
                feedback4 = 0.0

            self.rot_feedback = [feedback3, feedback4]
            self.gesture_feedback = None

            self.scene_anchor_save = [*self.panda.get_position(), *self.panda.get_orientation(scalar_first=False)]
            self.is_drawing = False


def main(args):
    import threading
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    import tkinter as tk
    import time
    import math
    from time import perf_counter, process_time

    class Visualizer:
        def __init__(self, width=800, height=400, title="Visualizer"):
            self.width = int(width)
            self.height = int(height)

            self.root = tk.Tk()
            self.root.title(title)
            self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="white")
            self.canvas.pack()

            self.point = self.plot_visible_point(0.5, 0.0)

        def _clamp(self, v, lo, hi):
            return lo if v < lo else hi if v > hi else v

        def _map_x(self, x):
            x = self._clamp(float(x), 0.0, 1.0)
            return x * (self.width - 1)

        def _map_y(self, y):
            y = self._clamp(float(y), -0.4, 0.4)
            t = (y + 0.4) / 0.8  # [-0.4,0.4] -> [0,1]
            return (1.0 - t) * (self.height - 1)

        def plot_visible_point(self, x, y, radius=3):
            cx = self._map_x(x)
            cy = self._map_y(y)
            r = float(radius)
            return self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="black", outline="")

        def update_visible_point(self, handle, x, y, radius=3):
            cx = self._map_x(x)
            cy = self._map_y(y)
            r = float(radius)
            self.canvas.coords(handle, cx - r, cy - r, cx + r, cy + r)

        def after(self, ms, fn):
            self.root.after(ms, fn)

        def run(self):
            self.root.mainloop()

    class FakePanda():
        def get_position(self):
            return [0.4,0.0,0.4]

        def get_orientation(self, scalar_first):
            return [1.0,0.0,0.0,0.0]

    rclpy.init()

    class SpinningRosNode(Node):
        def __init__(self):
            super().__init__(f"panda_node_{np.random.randint(100000)}")

            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self)

            self._spin_thread = threading.Thread(target=self._spin, daemon=True)
            self._spin_thread.start()

        def _spin(self):
            # Small timeout makes the thread yield regularly.
            # 0.01 is a good starting point.
            while rclpy.ok():
                self._executor.spin_once(timeout_sec=0.01)


    class TeleoperationByDrawingNode(TeleoperationByDrawing, SpinningRosNode):
        pass

    teleop = TeleoperationByDrawingNode(
        teleop_hand = args['teleop_hand'], 
        teleop_aux_hand = args['teleop_aux_hand'],
        link_gesture = args['link_gesture'],
        teleop_scale = args['teleop_scale'],
        # teleop_rotate_eef = args['teleop_rotate_eef'],
    )
    teleop.panda = FakePanda()
    teleop.viz = Visualizer()
    teleop.super_gp = [0.4,0.0,0.4] 
    teleop.feedback = [0.0,0.0,0.0,0.0,0.0]
    teleop.feedback_gripper = "none"
    times = [0.001]


    GUI_HZ = 60
    TELEOP_HZ = 20

    last_print = (0.0, 0.0)

    def tick():
        nonlocal last_print
        t0 = perf_counter(), process_time()

        teleop.teleop_step()
        t1 = perf_counter(), process_time()

        # GUI update (fast; one coords call)
        teleop.viz.update_visible_point(teleop.viz.point, teleop.t/1000, teleop.t/1000)
        t2 = perf_counter(), process_time()

        # throttle prints (e.g., 10 Hz)
        now = perf_counter(), process_time()
        if now[0] - last_print[0] > 0.1:
            print(f"wall: tick {(t2[0]-t0[0])*1000:.2f}ms (teleop {(t1[0]-t0[0])*1000:.2f}ms, gui {(t2[0]-t1[0])*1000:.2f}ms); cpu: tick {(t2[1]-t0[1])*1000:.2f}ms (teleop {(t1[1]-t0[1])*1000:.2f}ms, gui {(t2[1]-t1[1])*1000:.2f}ms)")
            last_print = now

        teleop.viz.root.after(int(1000 / TELEOP_HZ), tick)

    teleop.viz.root.after(0, tick)
    teleop.viz.run()


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
