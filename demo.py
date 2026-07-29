#!/usr/bin/env python3
"""Runnable demonstration of the hand-tracking control pipeline.

    python demo.py            # synthetic hand poses, no camera needed
    python demo.py --live     # real webcam + MediaPipe, if installed

The synthetic path exercises the whole chain — landmarks → flexion angles →
calibrated motor positions → atomic command file — so the repo can be verified
without a camera, MediaPipe, or a robot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.bridge import CommandWriter, read_command
from src.landmarks import FINGER_NAMES, FINGERS, N_LANDMARKS, all_flexions
from src.mapping import DEFAULT_CALIBRATION, HandController


def synthetic_hand(curl: float) -> np.ndarray:
    """Generate landmarks for a hand at a given curl, 0.0 (open) to 1.0 (fist).

    Each finger is built as a chain of segments; `curl` bends the PIP and DIP
    joints progressively, which is what the flexion calculation measures.
    """
    lm = np.zeros((N_LANDMARKS, 3))
    lm[0] = [0.4, 0.0, 0.0]                       # wrist

    for i, (name, (mcp, pip, dip, tip)) in enumerate(FINGERS.items()):
        x = i * 0.2
        seg = 0.25
        theta = curl * np.pi / 2                  # bend angle per joint

        lm[mcp] = [x, 0.30, 0.0]
        lm[pip] = [x, 0.30 + seg, 0.0]
        # each subsequent segment rotates by theta in the y-z plane
        lm[dip] = lm[pip] + [0.0, seg * np.cos(theta), seg * np.sin(theta)]
        lm[tip] = lm[dip] + [0.0, seg * np.cos(2 * theta), seg * np.sin(2 * theta)]
    return lm


def run_synthetic() -> None:
    print("Vision-controlled prosthetic hand — synthetic verification")
    print("(no camera or MediaPipe required)\n")

    ctrl = HandController(deadband=8, smoothing=0.5)
    out_path = Path("/tmp/_hand_demo/commands.txt")
    writer = CommandWriter(out_path)

    print("  Calibration in use")
    for name in FINGER_NAMES:
        c = DEFAULT_CALIBRATION[name]
        print(f"    {name:<7} flexion {c.open_angle:5.1f}–{c.closed_angle:5.1f}°  "
              f"→ motor {c.motor_open}–{c.motor_closed}")

    print(f"\n  {'curl':>5}  {'index flexion':>14}  {'motor commands sent':>44}")
    print("  " + "-" * 70)

    total_sent = 0
    for curl in np.linspace(0.0, 1.0, 9):
        lm = synthetic_hand(float(curl))
        flex = all_flexions(lm)
        cmds = ctrl.update(flex)
        total_sent += len(cmds)
        if cmds:
            frame = writer.send(cmds)
            f, parsed = read_command(out_path)
            assert f == frame and parsed == cmds, "command file round-trip failed"
        shown = ", ".join(f"{k}={v}" for k, v in sorted(cmds.items())) or "(within deadband)"
        print(f"  {curl:5.2f}  {flex['index']:11.1f}°  {shown:>44}")

    print(f"\n  {total_sent} motor commands issued, every one verified "
          f"complete after write")

    # deadband behaviour
    print("\n  Deadband — repeated near-identical poses must not re-command")
    ctrl.reset()
    ctrl.update(all_flexions(synthetic_hand(0.5)))
    repeats = sum(len(ctrl.update(all_flexions(synthetic_hand(0.5 + j * 0.001))))
                  for j in range(10))
    print(f"    10 near-identical frames → {repeats} commands "
          f"({'suppressed correctly' if repeats == 0 else 'CHATTER — BUG'})")

    # clamping
    print("\n  Clamping — flexion beyond calibration must not exceed motor limits")
    extreme = {n: 1e6 for n in FINGER_NAMES}
    ctrl.reset()
    cmds = ctrl.update(extreme)
    ok = all(
        min(DEFAULT_CALIBRATION[n].motor_open, DEFAULT_CALIBRATION[n].motor_closed)
        <= v <=
        max(DEFAULT_CALIBRATION[n].motor_open, DEFAULT_CALIBRATION[n].motor_closed)
        for n, v in cmds.items()
    )
    print(f"    flexion = 1e6° → {cmds}")
    print(f"    all within mechanical limits: {ok}")

    # scale invariance
    print("\n  Scale invariance — the reason we use angles, not coordinates")
    near = synthetic_hand(0.6)
    far = near * 0.35                       # same pose, further from camera
    a, b = all_flexions(near)["index"], all_flexions(far)["index"]
    print(f"    hand near camera: {a:.2f}°   hand far away: {b:.2f}°   "
          f"difference {abs(a - b):.2e}°")


def run_live() -> None:
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        sys.exit("--live needs opencv-python and mediapipe:\n"
                 "    pip install opencv-python mediapipe")

    ctrl = HandController()
    writer = CommandWriter(Path("commands.txt"))
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
    cap = cv2.VideoCapture(0)
    print("Live mode — press q to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                pts = np.array([[p.x, p.y, p.z]
                                for p in result.multi_hand_landmarks[0].landmark])
                cmds = ctrl.update(all_flexions(pts))
                if cmds:
                    writer.send(cmds)
                    print(f"  {cmds}")
            cv2.imshow("hand", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use a real webcam + MediaPipe")
    args = ap.parse_args()
    run_live() if args.live else run_synthetic()
