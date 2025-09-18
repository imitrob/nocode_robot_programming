import time, threading, os, errno
from dataclasses import dataclass, field
from typing import Dict, Optional
from evdev import InputDevice, ecodes
from spatialmath import UnitQuaternion
import spatialmath as sm 

# >>>>>> EDIT THIS to your by-id symlink (from ls -l /dev/input/by-id)
DEVICE_PATH = "/dev/input/by-id/usb-Logitech_Wireless_Gamepad_F710_B62E92A9-event-joystick"
# DEVICE_PATH = "/dev/input/by-id/usb-Logitech_Logitech_Cordless_RumblePad_2-event-joystick"

BUTTONS = {
    ecodes.BTN_A:     "A",
    ecodes.BTN_B:     "B",
    ecodes.BTN_X:     "X",
    ecodes.BTN_Y:     "Y",
    ecodes.BTN_TL:    "LB",
    ecodes.BTN_TR:    "RB",
    ecodes.BTN_THUMBL:"L_STICK",
    ecodes.BTN_THUMBR:"R_STICK",
    ecodes.BTN_START: "START",
    ecodes.BTN_SELECT:"BACK",
    ecodes.BTN_MODE:  "GUIDE",
}

AXES = {
    ecodes.ABS_X:   "LX",
    ecodes.ABS_Y:   "LY",
    ecodes.ABS_Z:  "RX",
    ecodes.ABS_RZ:  "RY",
    # ecodes.ABS_Z:   "LT",        # trigger
    # ecodes.ABS_RZ:  "RT",        # trigger
    ecodes.ABS_HAT0X:"HAT_X",    # d-pad
    ecodes.ABS_HAT0Y:"HAT_Y",
}

FREQ = 20 # Hz

@dataclass
class GamepadState:
    buttons: Dict[str, int] = field(default_factory=dict)
    axes: Dict[str, float] = field(default_factory=dict)
    raw_axes: Dict[str, int] = field(default_factory=dict)

class JoystickConnector():
    def __init__(self):
        super(JoystickConnector, self).__init__()
        self.joy_path = DEVICE_PATH
        self.joy_state = GamepadState()
        self._joy_stop = threading.Event()

        self.joy_period = 1/FREQ
        self.joy_last_clicked = False

        self._joy_thread_control = None

    def _joy_open(self) -> Optional[InputDevice]:
        try:
            return InputDevice(self.joy_path)
        except FileNotFoundError:
            return None

    def _joy_listener_run(self):
        dev = None
        absinfo_cache = {}
        while not self._joy_stop.is_set():
            if dev is None:
                dev = self._joy_open()
                if dev is None:
                    time.sleep(0.5)
                    continue

                # Build a map {abs_code: AbsInfo}
                caps = dev.capabilities(absinfo=True)
                abs_caps = dict(caps.get(ecodes.EV_ABS, []))  # list[(code, AbsInfo)] -> dict

                # AXES is likely {code: "name"}; keep only the ones we care about
                absinfo_cache = {code: abs_caps.get(code) for code in AXES.keys()}

                print(f"[reader] connected: {dev.path} - {dev.name}")

            try:
                for event in dev.read_loop():
                    if self._joy_stop.is_set():
                        break

                    if event.type == ecodes.EV_KEY and event.code in BUTTONS:
                        self.joy_state.buttons[BUTTONS[event.code]] = int(event.value)

                    elif event.type == ecodes.EV_ABS and event.code in AXES:
                        name = AXES[event.code]
                        raw = event.value
                        self.joy_state.raw_axes[name] = raw

                        # Get AbsInfo, refresh cache lazily if missing
                        ai = absinfo_cache.get(event.code)
                        if ai is None:
                            ai = dev.absinfo(event.code)
                            absinfo_cache[event.code] = ai  # cache for next time

                        if name in ("LX", "LY", "RX", "RY"):
                            if ai and (ai.max != ai.min):
                                val = ((raw - ai.min) / (ai.max - ai.min)) * 2.0 - 1.0
                            else:
                                val = float(raw)  # fallback if device reports no range

                            if name in ("LY", "RY"):  # invert Y so up = +1
                                val = -val
                            if abs(val) < 0.05:
                                val = 0.0
                            self.joy_state.axes[name] = val

                        elif name in ("LT", "RT"):
                            if ai and (ai.max != ai.min):
                                self.joy_state.axes[name] = (raw - ai.min) / (ai.max - ai.min)
                            else:
                                self.joy_state.axes[name] = float(raw)

                        elif name in ("HAT_X", "HAT_Y"):
                            self.joy_state.axes[name] = float(raw)

            except OSError as e:
                if e.errno in (errno.ENODEV, errno.EIO):
                    print("[reader] device lost; waiting to reconnect…")
                    try:
                        dev.close()
                    except Exception:
                        pass
                    dev = None
                    absinfo_cache = {}  # drop cache so we rebuild after reconnect
                    time.sleep(0.5)
                    continue
                else:
                    raise




    def joy_stop(self):
        self._joy_stop.set()
        self._joy_thread_listener.join(timeout=1)
        self._joy_thread_control.join(timeout=1)

    def joy_start(self):
        self._joy_stop.clear()
        self._joy_thread_listener = threading.Thread(target=self._joy_listener_run, daemon=True)
        self._joy_thread_listener.start()
        self._joy_thread_control = threading.Thread(target=self._joy_thread_start, daemon=True)
        self._joy_thread_control.start()

    def _joy_thread_start(self):
        try:
            while not self._joy_stop.is_set():
                self.joy_step()
                time.sleep(self.joy_period)
        except KeyboardInterrupt:
            pass

    def joy_step(self):
        s = self.joy_state
        
        self.feedback[1] = round(s.axes.get("LX", 0.0),1) *  self.feedback_gain  # left stick X
        self.feedback[0] = -round(s.axes.get("LY", 0.0),1) * self.feedback_gain  # left stick Y
        self.feedback[2] = round(s.axes.get("RY", 0.0),1) *  self.feedback_gain  # right stick Y
        eef_rot = round(s.axes.get("RX", 0.0),1)  *          self.feedback_gain  # right stick X

        q = UnitQuaternion([0.0,1.0,0.0,0.0])
        rot = sm.SO3(q.R) * sm.SO3.Rz(eef_rot)
        self.feedback[3],self.feedback[4],self.feedback[5],self.feedback[6] = UnitQuaternion(rot).vec_xyzs

        # print("Joystick joy_state:", self.feedback)

        if s.buttons.get("A", 0) > 0:
            if not self.gripper.read_once().is_grasped:
                self.feedback_gripper = "grasp"
        elif s.buttons.get("B", 0) > 0:
            self.feedback_gripper = "open"
            
        if s.buttons.get("X", 0) > 0:
            self.pause = True


if __name__ == "__main__":
    from robot import PandaPy
    class FrankaJoystick(PandaPy, JoystickConnector):
        '''
        Panda:
            self.move_to_pose(position, orientation) # position (float[3]), orientation (float[4])
            self.grasp(width, speed, force, epsilon_inner, epsilon_outer)
        '''
        def __init__(self):
            super(FrankaJoystick, self).__init__()

    r = FrankaJoystick()
    r.move_to_pose(position=[0.4,0.0,0.4], orientation=[1.0,0.0,0.0,0.0])
    r.joy_start()
    try:
        while True:
            s = r.joy_state
            print({
                "A": s.buttons.get("A", 0),
                "B": s.buttons.get("B", 0),
                "LX": round(s.axes.get("LX", 0.0), 3),
                "LY": round(s.axes.get("LY", 0.0), 3),
                "RX": round(s.axes.get("RX", 0.0), 3),
                "RY": round(s.axes.get("RY", 0.0), 3),
                "LT": round(s.axes.get("LT", 0.0), 3),
                "RT": round(s.axes.get("RT", 0.0), 3),
                "DPAD": (int(s.axes.get("HAT_X", 0)), int(s.axes.get("HAT_Y", 0))),
            })
            time.sleep(1./FREQ)

            r.move_to_pose(position=r.feedback[:3], orientation=r.feedback[3:])
    except KeyboardInterrupt:
        r.stop()

