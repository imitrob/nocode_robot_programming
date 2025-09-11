import time, threading, os, errno
from dataclasses import dataclass, field
from typing import Dict, Optional
from evdev import InputDevice, ecodes
from spatialmath import UnitQuaternion
import spatialmath as sm 

# >>>>>> EDIT THIS to your by-id symlink (from ls -l /dev/input/by-id)
DEVICE_PATH = "/dev/input/by-id/usb-Logitech_Wireless_Gamepad_F710_B62E92A9-event-joystick"

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
    ecodes.ABS_RX:  "RX",
    ecodes.ABS_RY:  "RY",
    ecodes.ABS_Z:   "LT",        # trigger
    ecodes.ABS_RZ:  "RT",        # trigger
    ecodes.ABS_HAT0X:"HAT_X",    # d-pad
    ecodes.ABS_HAT0Y:"HAT_Y",
}

@dataclass
class GamepadState:
    buttons: Dict[str, int] = field(default_factory=dict)
    axes: Dict[str, float] = field(default_factory=dict)
    raw_axes: Dict[str, int] = field(default_factory=dict)

class JoystickReader:
    def __init__(self):
        self.path = DEVICE_PATH
        self.state = GamepadState()
        self._stop = threading.Event()
        self._thread_listener = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread_listener.start()

    def stop(self):
        self._stop.set()
        self._thread_listener.join(timeout=1)

    def _open(self) -> Optional[InputDevice]:
        try:
            return InputDevice(self.path)
        except FileNotFoundError:
            return None


    def _run(self):
        dev = None
        absinfo_cache = {}
        while not self._stop.is_set():
            if dev is None:
                dev = self._open()
                if dev is None:
                    time.sleep(0.5)
                    continue
                try:
                    # cache abs ranges for normalization
                    absinfo_cache = {code: dev.absinfo(code) for code in AXES if code in dev.capabilities().get(ecodes.EV_ABS, [])}
                    print(f"[reader] connected: {dev.path} - {dev.name}")
                except Exception:
                    pass

            try:
                for event in dev.read_loop():
                    if self._stop.is_set():
                        break
                    if event.type == ecodes.EV_KEY and event.code in BUTTONS:
                        self.state.buttons[BUTTONS[event.code]] = int(event.value)
                    elif event.type == ecodes.EV_ABS and event.code in AXES:
                        name = AXES[event.code]
                        raw = event.value
                        self.state.raw_axes[name] = raw
                        ai = absinfo_cache.get(event.code)
                        if name in ("LX","LY","RX","RY"):
                            if ai and (ai.max - ai.min):
                                val = ((raw - ai.min) / (ai.max - ai.min)) * 2.0 - 1.0
                            else:
                                val = float(raw)
                            if name in ("LY","RY"):  # invert Y to up=+1
                                val = -val
                            if abs(val) < 0.05:
                                val = 0.0
                            self.state.axes[name] = val
                        elif name in ("LT","RT"):
                            if ai and (ai.max - ai.min):
                                self.state.axes[name] = (raw - ai.min) / (ai.max - ai.min)
                            else:
                                self.state.axes[name] = float(raw)
                        elif name in ("HAT_X","HAT_Y"):
                            self.state.axes[name] = float(raw)
            except OSError as e:
                # 19 = ENODEV -> device gone (sleep/replug); 5 = EIO sometimes on unplug
                if e.errno in (errno.ENODEV, errno.EIO):
                    print("[reader] device lost; waiting to reconnect…")
                    try:
                        dev.close()
                    except Exception:
                        pass
                    dev = None
                    time.sleep(0.5)
                    continue
                else:
                    raise

class JoystickControl(JoystickReader):
    def __init__(self, node):
        super(JoystickControl, self).__init__()

        self.tgoal_pose = [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        self.eef_rot = 0.0
        self.node = node
        self.period = 1/10

        self.start()

        self.last_clicked = False

    def start(self):
        self._thread_listener.start()

    def stop(self):
        self._stop.set()
        self._thread_listener.join(timeout=1)

    def step(self):
        s = self.state

        
        self.tgoal_pose[1] += round(s.axes.get("LX", 0.0) * (1/2**15),1) * 0.005  # left stick X
        self.tgoal_pose[0] += -round(s.axes.get("LY", 0.0) * (1/2**15),1) * 0.005  # left stick Y
        self.tgoal_pose[2] += round(s.axes.get("RY", 0.0) * (1/2**15),1) * 0.005  # right stick Y
        self.eef_rot += round(s.axes.get("RX", 0.0) * (1/2**15),1)  * 0.05          # right stick X

        q = UnitQuaternion([0.0,1.0,0.0,0.0])
        rot = sm.SO3(q.R) * sm.SO3.Rz(self.eef_rot)
        self.tgoal_pose[3],self.tgoal_pose[4],self.tgoal_pose[5],self.tgoal_pose[6] = UnitQuaternion(rot).vec_xyzs

        print("Joystick state:", self.tgoal_pose)
        # print({
        #     "A": s.buttons.get("A", 0),
        #     "B": s.buttons.get("B", 0),
        #     "LX": round(s.axes.get("LX", 0.0), 3),
        #     "LY": round(s.axes.get("LY", 0.0), 3),
        #     "RX": round(s.axes.get("RX", 0.0), 3),
        #     "RY": round(s.axes.get("RY", 0.0), 3),
        #     "LT": round(s.axes.get("LT", 0.0), 3),
        #     "RT": round(s.axes.get("RT", 0.0), 3),
        #     "DPAD": (int(s.axes.get("HAT_X", 0)), int(s.axes.get("HAT_Y", 0))),
        # })
        self.node.move_to_pose(self.tgoal_pose[0:3], self.tgoal_pose[3:7], 1.0)

        if s.buttons.get("A", 0) > 0:
            if not self.node.gripper.read_once().is_grasped:
                self.node.gripper.grasp(width=0, speed=0.2, force=10, epsilon_inner=0.04, epsilon_outer=0.04)
        elif s.buttons.get("B", 0) > 0:
            self.node.gripper.move(0.08, 0.2)

        if s.buttons.get("X", 0) > 0:
            self.node.risk_flag = 1
            self.last_clicked = True
        if s.buttons.get("X", 0) == 0 and self.last_clicked == True:
            self.node.risk_flag = 0

        time.sleep(self.period)
        


if __name__ == "__main__":
    r = JoystickReader()
    r.start()
    try:
        # sample at 50 Hz
        period = 1/50
        while True:
            s = r.state
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
            time.sleep(period)
    except KeyboardInterrupt:
        r.stop()

