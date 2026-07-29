"""Mapping finger flexion angles to Dynamixel AX-12A motor positions.

The AX-12A takes a goal position of 0-1023 over a 300 degree span, so one unit
is roughly 0.29 degrees.

Three things stand between "angle computed" and "motor moves correctly":

**Per-finger calibration.** Nobody's hand has the same range of motion, and no
two fingers on the same hand do either. Each finger is calibrated by recording
the flexion angle fully open and fully closed, then normalised against that
range. Without it a user with less mobility never reaches full grip.

**Mechanical limits.** The prosthetic's fingers physically cannot travel the
motor's whole range — driving past the limit stalls the servo, which draws
stall current and heats it. Every command is clamped to per-motor limits, and
the clamp is tested.

**A deadband.** Landmark jitter of a degree or two produces a constant stream
of tiny position changes. The motors chatter audibly and the gesture looks
nervous. Commands are only re-sent when the change exceeds a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# AX-12A specification
POSITION_MIN = 0
POSITION_MAX = 1023
DEGREES_PER_UNIT = 300.0 / 1023.0


@dataclass
class FingerCalibration:
    """Flexion angles observed at the extremes of one finger's travel."""

    open_angle: float = 0.0     # degrees of flexion with the finger straight
    closed_angle: float = 180.0  # degrees of flexion in a full fist
    motor_open: int = 200        # motor position corresponding to open
    motor_closed: int = 800      # motor position corresponding to closed
    inverted: bool = False       # some motors are mounted mirrored

    def __post_init__(self) -> None:
        if self.closed_angle <= self.open_angle:
            raise ValueError(
                f"closed_angle ({self.closed_angle}) must exceed "
                f"open_angle ({self.open_angle})"
            )
        for name, v in (("motor_open", self.motor_open), ("motor_closed", self.motor_closed)):
            if not POSITION_MIN <= v <= POSITION_MAX:
                raise ValueError(f"{name}={v} outside AX-12A range 0-1023")

    @property
    def span(self) -> float:
        return self.closed_angle - self.open_angle


DEFAULT_CALIBRATION: dict[str, FingerCalibration] = {
    "thumb":  FingerCalibration(0.0, 120.0, 250, 700),
    "index":  FingerCalibration(0.0, 180.0, 200, 800),
    "middle": FingerCalibration(0.0, 190.0, 200, 820),
    "ring":   FingerCalibration(0.0, 185.0, 210, 810),
    "pinky":  FingerCalibration(0.0, 170.0, 220, 780),
}


def normalise_flexion(angle: float, cal: FingerCalibration) -> float:
    """Flexion angle -> 0.0 (fully open) .. 1.0 (fully closed), clamped."""
    return float(np.clip((angle - cal.open_angle) / cal.span, 0.0, 1.0))


def flexion_to_position(angle: float, cal: FingerCalibration) -> int:
    """Flexion angle -> AX-12A goal position, clamped to the motor's limits."""
    t = normalise_flexion(angle, cal)
    if cal.inverted:
        t = 1.0 - t
    pos = cal.motor_open + t * (cal.motor_closed - cal.motor_open)

    lo, hi = sorted((cal.motor_open, cal.motor_closed))
    return int(round(float(np.clip(pos, lo, hi))))


def position_to_degrees(position: int) -> float:
    """AX-12A units -> degrees of shaft rotation."""
    return float(position) * DEGREES_PER_UNIT


@dataclass
class HandController:
    """Turns per-frame flexion angles into motor commands.

    Holds the calibration, the deadband, and the last commanded position for
    each finger, so it can decide what actually needs sending.
    """

    calibration: dict[str, FingerCalibration] = field(
        default_factory=lambda: dict(DEFAULT_CALIBRATION)
    )
    deadband: int = 8          # AX-12A units (~2.3 degrees) before re-commanding
    smoothing: float = 0.4     # EMA weight on the new sample
    _last_sent: dict[str, int] = field(default_factory=dict, init=False)
    _smoothed: dict[str, float] = field(default_factory=dict, init=False)

    def update(self, flexions: dict[str, float]) -> dict[str, int]:
        """Return only the motor commands that changed beyond the deadband."""
        commands: dict[str, int] = {}

        for finger, angle in flexions.items():
            cal = self.calibration.get(finger)
            if cal is None:
                continue

            target = float(flexion_to_position(angle, cal))
            prev = self._smoothed.get(finger, target)
            smoothed = prev + self.smoothing * (target - prev)
            self._smoothed[finger] = smoothed

            pos = int(round(smoothed))
            last = self._last_sent.get(finger)
            if last is None or abs(pos - last) >= self.deadband:
                commands[finger] = pos
                self._last_sent[finger] = pos

        return commands

    def current_positions(self) -> dict[str, int]:
        return dict(self._last_sent)

    def reset(self) -> None:
        self._last_sent.clear()
        self._smoothed.clear()
