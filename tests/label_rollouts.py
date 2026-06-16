#!/usr/bin/env python3
"""
Interactive rollout labeler.

Plays each rollout in success_accuracy.py and lets you set its success value.

Keys during playback or on the end-of-video pause screen:
  1  — mark as success (1)
  0  — mark as failure (0)
  r  — replay from the beginning
  q  — save current state and quit

Progress is saved to success_accuracy.py after every annotation.
"""
import sys
import pprint
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2

from nocode_robot_programming.state_decision_dataset_prepare.dataloader import TrajectoryDataset
from nocode_robot_programming.state_decision_dataset_prepare.success_accuracy import (
    per_rollout as per_rollout_orig,
)

TRAJECTORIES_DIR = Path(TrajectoryDataset().dir)
SUCCESS_ACCURACY_PATH = (
    Path(__file__).resolve().parents[1]
    / "nocode_robot_programming"
    / "state_decision_dataset_prepare"
    / "success_accuracy.py"
)

FPS = 50  # 20 fps * 2.5x playback speed
DELAY_MS = max(1, 1000 // FPS)
SCALE = 2.5  # upscale factor for display (180x320 → 450x800)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_frames(rollout_name: str):
    path = TRAJECTORIES_DIR / f"{rollout_name}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False, mmap_mode="r")
    return data["img"]  # (N, H, W), uint8


def play_rollout(rollout_name: str, frames: np.ndarray, current_val: int, idx: int, total: int) -> str:
    """
    Play frames in a cv2 window.
    Returns the key pressed: '1', '0', 'r', or 'q'.
    """
    h, w = frames.shape[1], frames.shape[2]
    dh, dw = int(h * SCALE), int(w * SCALE)
    win_name = "Rollout labeler"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, dw, dh + 60)

    DECISION_KEYS = {ord("1"), ord("0"), ord("r"), ord("q")}

    def render(frame_bgr: np.ndarray, footer: str) -> np.ndarray:
        canvas = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_NEAREST)
        # Dark header strip
        header = np.zeros((40, dw, 3), dtype=np.uint8)
        cv2.putText(header, f"[{idx}/{total}]  {rollout_name}  (current: {current_val})",
                    (6, 27), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        # Dark footer strip
        foot = np.zeros((40, dw, 3), dtype=np.uint8)
        cv2.putText(foot, footer, (6, 27), FONT, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
        return np.vstack([header, canvas, foot])

    while True:
        # --- playback loop ---
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            img = render(bgr, "1=success   0=failure   r=replay   q=quit")
            cv2.imshow(win_name, img)
            key = cv2.waitKey(DELAY_MS) & 0xFF
            if key in DECISION_KEYS:
                cv2.destroyAllWindows()
                return chr(key)

        # --- end-of-video: freeze on last frame and wait ---
        bgr = cv2.cvtColor(frames[-1], cv2.COLOR_GRAY2BGR)
        img = render(bgr, "END  —  1=success   0=failure   r=replay   q=quit")
        cv2.imshow(win_name, img)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("1"), ord("0"), ord("q")):
                cv2.destroyAllWindows()
                return chr(key)
            if key == ord("r"):
                break  # restart play loop


def save(data: dict) -> None:
    content = "per_rollout = " + pprint.pformat(data, width=120) + "\n"
    SUCCESS_ACCURACY_PATH.write_text(content)


def main() -> None:
    # Deep-copy to avoid mutating the imported module's dict
    updated = {
        user: {
            task: {
                modality: [dict(ds_dict) for ds_dict in ds_list]
                for modality, ds_list in modalities.items()
            }
            for task, modalities in tasks.items()
        }
        for user, tasks in per_rollout_orig.items()
    }

    # Flat ordered list of all rollout entries
    entries = [
        (user, task, modality, ds_idx, rollout_name)
        for user, tasks in updated.items()
        for task, modalities in tasks.items()
        for modality, ds_list in modalities.items()
        for ds_idx, ds_dict in enumerate(ds_list)
        for rollout_name in ds_dict
    ]

    total = len(entries)
    print(f"Loaded {total} rollout entries from success_accuracy.py")
    print("Window controls:  1 = success   0 = failure   r = replay   q = save & quit\n")

    for i, (user, task, modality, ds_idx, rollout_name) in enumerate(entries):
        current = updated[user][task][modality][ds_idx][rollout_name]
        print(f"[{i+1}/{total}]  {rollout_name}  (current={current})")

        frames = load_frames(rollout_name)
        if frames is None:
            print(f"  WARNING: file not found in {TRAJECTORIES_DIR}, skipping")
            continue

        while True:
            key = play_rollout(rollout_name, frames, current, i + 1, total)

            if key == "r":
                continue
            if key == "q":
                print("Saving and quitting…")
                save(updated)
                sys.exit(0)

            val = int(key)
            updated[user][task][modality][ds_idx][rollout_name] = val
            print(f"  → {val}")
            break

        save(updated)

    print(f"\nAll {total} rollouts labeled. success_accuracy.py updated.")


if __name__ == "__main__":
    main()
