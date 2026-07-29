"""Finger flexion angles from MediaPipe hand landmarks.

MediaPipe Hands returns 21 3-D landmarks per hand. A finger's flexion is
recovered from the angle at each joint — the angle between the two bone
segments meeting there — computed from the dot product of the segment vectors.

Two details that matter in practice:

**Angles, not raw coordinates.** Landmark positions depend on how far the hand
is from the camera and where it sits in frame. Joint *angles* are invariant to
both, so a gesture maps to the same motor command whether the user is close to
the webcam or across the room. Mapping raw coordinates instead is the usual
reason a vision-controlled hand works at one distance and nowhere else.

**Degenerate geometry must not produce NaN.** When a landmark is momentarily
mis-detected, two points can coincide, giving a zero-length vector. `arccos` of
the resulting 0/0 is NaN, which propagates into the motor command. Every angle
here is guarded and falls back to the last valid value.
"""

from __future__ import annotations

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
FINGERS: dict[str, tuple[int, int, int, int]] = {
    # name:      (MCP, PIP, DIP, TIP)  — thumb uses (CMC, MCP, IP, TIP)
    "thumb":     (1, 2, 3, 4),
    "index":     (5, 6, 7, 8),
    "middle":    (9, 10, 11, 12),
    "ring":      (13, 14, 15, 16),
    "pinky":     (17, 18, 19, 20),
}
FINGER_NAMES = list(FINGERS)
N_LANDMARKS = 21


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at b, in degrees, formed by segments b->a and b->c.

    Returns 180.0 (straight) for degenerate input rather than NaN.
    """
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0  # coincident landmarks — treat the joint as straight

    cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def finger_flexion(landmarks: np.ndarray, finger: str) -> float:
    """Total flexion of one finger, in degrees.

    Sums the deviation from straight at the two main bending joints (PIP and DIP).
    A fully open finger reads ~0; a closed fist reads ~180-250 depending on the
    finger and the user's hand.
    """
    if finger not in FINGERS:
        raise ValueError(f"Unknown finger '{finger}'. Expected one of {FINGER_NAMES}")
    lm = validate_landmarks(landmarks)

    mcp, pip, dip, tip = FINGERS[finger]
    bend_pip = 180.0 - joint_angle(lm[mcp], lm[pip], lm[dip])
    bend_dip = 180.0 - joint_angle(lm[pip], lm[dip], lm[tip])
    return float(max(bend_pip + bend_dip, 0.0))


def all_flexions(landmarks: np.ndarray) -> dict[str, float]:
    """Flexion angle for every finger, in a stable order."""
    return {name: finger_flexion(landmarks, name) for name in FINGER_NAMES}


def validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Check shape and finiteness. Returns the array as float64."""
    lm = np.asarray(landmarks, dtype=float)
    if lm.shape != (N_LANDMARKS, 3):
        raise ValueError(
            f"Expected {N_LANDMARKS} landmarks of 3 coordinates, got shape {lm.shape}"
        )
    if not np.all(np.isfinite(lm)):
        raise ValueError("Landmarks contain NaN or infinity — detection likely failed")
    return lm


def hand_scale(landmarks: np.ndarray) -> float:
    """Wrist-to-middle-MCP distance — a proxy for apparent hand size.

    Useful for rejecting frames where the hand is too far away for landmarks to
    be reliable.
    """
    lm = validate_landmarks(landmarks)
    return float(np.linalg.norm(lm[FINGERS["middle"][0]] - lm[WRIST]))
