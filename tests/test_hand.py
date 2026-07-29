"""Tests for landmark geometry, motor mapping and the command bridge.

Runs without MediaPipe, a webcam, or a robot — the geometry and mapping are
pure functions over arrays, which is exactly why they're separated from the
capture loop.

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bridge import (  # noqa: E402
    FINGER_ORDER,
    CommandWriter,
    format_command,
    parse_command,
    read_command,
    write_atomic,
)
from src.landmarks import (  # noqa: E402
    FINGER_NAMES,
    N_LANDMARKS,
    all_flexions,
    finger_flexion,
    hand_scale,
    joint_angle,
    validate_landmarks,
)
from src.mapping import (  # noqa: E402
    DEFAULT_CALIBRATION,
    POSITION_MAX,
    POSITION_MIN,
    FingerCalibration,
    HandController,
    flexion_to_position,
    normalise_flexion,
    position_to_degrees,
)


def open_hand() -> np.ndarray:
    """Landmarks for a flat hand: every finger straight along +y."""
    lm = np.zeros((N_LANDMARKS, 3))
    from src.landmarks import FINGERS
    for i, (name, (mcp, pip, dip, tip)) in enumerate(FINGERS.items()):
        x = i * 0.2
        for k, idx in enumerate((mcp, pip, dip, tip)):
            lm[idx] = [x, 0.3 + 0.25 * k, 0.0]   # collinear => straight
    lm[0] = [0.4, 0.0, 0.0]                      # wrist
    return lm


def curled_hand() -> np.ndarray:
    """Landmarks with each finger bent sharply at PIP and DIP."""
    lm = np.zeros((N_LANDMARKS, 3))
    from src.landmarks import FINGERS
    for i, (name, (mcp, pip, dip, tip)) in enumerate(FINGERS.items()):
        x = i * 0.2
        lm[mcp] = [x, 0.30, 0.0]
        lm[pip] = [x, 0.55, 0.0]
        lm[dip] = [x, 0.55, 0.25]    # 90 degree bend
        lm[tip] = [x, 0.30, 0.25]    # another 90 degree bend
    lm[0] = [0.4, 0.0, 0.0]
    return lm


# ==========================================================================
# Geometry
# ==========================================================================
class TestJointAngle:
    def test_straight_is_180_degrees(self):
        a, b, c = np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([2, 0, 0])
        assert joint_angle(a, b, c) == pytest.approx(180.0)

    def test_right_angle(self):
        a, b, c = np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])
        assert joint_angle(a, b, c) == pytest.approx(90.0)

    def test_folded_back_is_zero(self):
        a, b, c = np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 0])
        assert joint_angle(a, b, c) == pytest.approx(0.0)

    def test_coincident_points_do_not_produce_nan(self):
        """A mis-detected landmark can duplicate another — must not give NaN."""
        p = np.array([1.0, 1.0, 1.0])
        angle = joint_angle(p, p, np.array([2.0, 2.0, 2.0]))
        assert np.isfinite(angle)
        assert angle == pytest.approx(180.0)

    def test_is_invariant_to_translation(self):
        a, b, c = np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])
        shift = np.array([100.0, -50.0, 7.0])
        assert joint_angle(a + shift, b + shift, c + shift) == pytest.approx(
            joint_angle(a, b, c)
        )

    def test_is_invariant_to_scale(self):
        """The reason we use angles: distance from the camera must not matter."""
        a, b, c = np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])
        assert joint_angle(a * 5, b * 5, c * 5) == pytest.approx(joint_angle(a, b, c))


class TestFingerFlexion:
    def test_straight_finger_reads_near_zero(self):
        for name in FINGER_NAMES:
            assert finger_flexion(open_hand(), name) == pytest.approx(0.0, abs=1e-6)

    def test_curled_finger_reads_high(self):
        for name in FINGER_NAMES:
            assert finger_flexion(curled_hand(), name) > 150.0

    def test_flexion_is_never_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            lm = rng.uniform(-1, 1, (N_LANDMARKS, 3))
            for name in FINGER_NAMES:
                assert finger_flexion(lm, name) >= 0.0

    def test_all_fingers_returned_in_stable_order(self):
        flex = all_flexions(open_hand())
        assert list(flex) == FINGER_NAMES

    def test_unknown_finger_rejected(self):
        with pytest.raises(ValueError, match="Unknown finger"):
            finger_flexion(open_hand(), "toe")


class TestValidation:
    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="Expected 21"):
            validate_landmarks(np.zeros((10, 3)))

    def test_nan_landmarks_rejected(self):
        lm = open_hand()
        lm[5, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or infinity"):
            validate_landmarks(lm)

    def test_hand_scale_is_positive(self):
        assert hand_scale(open_hand()) > 0


# ==========================================================================
# Motor mapping
# ==========================================================================
class TestCalibration:
    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="must exceed"):
            FingerCalibration(open_angle=100.0, closed_angle=50.0)

    def test_rejects_motor_position_out_of_range(self):
        with pytest.raises(ValueError, match="outside AX-12A range"):
            FingerCalibration(motor_open=-5)
        with pytest.raises(ValueError, match="outside AX-12A range"):
            FingerCalibration(motor_closed=2000)

    def test_defaults_cover_every_finger(self):
        assert set(DEFAULT_CALIBRATION) == set(FINGER_NAMES)


class TestFlexionMapping:
    def test_endpoints_map_to_endpoints(self):
        cal = FingerCalibration(0.0, 180.0, 200, 800)
        assert flexion_to_position(0.0, cal) == 200
        assert flexion_to_position(180.0, cal) == 800

    def test_midpoint_maps_to_midpoint(self):
        cal = FingerCalibration(0.0, 180.0, 200, 800)
        assert flexion_to_position(90.0, cal) == pytest.approx(500, abs=1)

    def test_beyond_range_is_clamped_not_extrapolated(self):
        """Driving past the mechanical limit stalls the servo."""
        cal = FingerCalibration(0.0, 180.0, 200, 800)
        assert flexion_to_position(-100.0, cal) == 200
        assert flexion_to_position(9999.0, cal) == 800

    def test_output_always_within_ax12a_range(self):
        rng = np.random.default_rng(1)
        for cal in DEFAULT_CALIBRATION.values():
            for angle in rng.uniform(-500, 500, 50):
                pos = flexion_to_position(float(angle), cal)
                assert POSITION_MIN <= pos <= POSITION_MAX

    def test_inverted_motor_reverses_direction(self):
        normal = FingerCalibration(0.0, 180.0, 200, 800, inverted=False)
        flipped = FingerCalibration(0.0, 180.0, 200, 800, inverted=True)
        assert flexion_to_position(0.0, normal) == flexion_to_position(180.0, flipped)

    def test_normalisation_is_bounded(self):
        cal = FingerCalibration(10.0, 150.0)
        assert normalise_flexion(-99.0, cal) == 0.0
        assert normalise_flexion(999.0, cal) == 1.0

    def test_calibration_adapts_to_reduced_range_of_motion(self):
        """A user who can only bend 90 degrees must still reach full grip."""
        limited = FingerCalibration(0.0, 90.0, 200, 800)
        assert flexion_to_position(90.0, limited) == 800

    def test_degrees_conversion(self):
        assert position_to_degrees(0) == pytest.approx(0.0)
        assert position_to_degrees(1023) == pytest.approx(300.0, abs=0.01)


class TestHandController:
    def test_first_update_commands_every_finger(self):
        ctrl = HandController()
        cmds = ctrl.update({n: 90.0 for n in FINGER_NAMES})
        assert set(cmds) == set(FINGER_NAMES)

    def test_deadband_suppresses_micro_jitter(self):
        """Landmark noise of a degree or two must not make the motors chatter."""
        ctrl = HandController(deadband=20, smoothing=1.0)
        ctrl.update({"index": 90.0})
        repeat = ctrl.update({"index": 90.3})
        assert "index" not in repeat

    def test_large_change_passes_the_deadband(self):
        ctrl = HandController(deadband=8, smoothing=1.0)
        ctrl.update({"index": 0.0})
        assert "index" in ctrl.update({"index": 180.0})

    def test_smoothing_damps_a_step_input(self):
        fast = HandController(deadband=0, smoothing=1.0)
        slow = HandController(deadband=0, smoothing=0.2)
        fast.update({"index": 0.0}); slow.update({"index": 0.0})
        f = fast.update({"index": 180.0})["index"]
        s = slow.update({"index": 180.0})["index"]
        assert s < f, "lower smoothing weight should move less per frame"

    def test_unknown_fingers_are_ignored(self):
        assert HandController().update({"tentacle": 90.0}) == {}

    def test_reset_clears_state(self):
        ctrl = HandController()
        ctrl.update({n: 90.0 for n in FINGER_NAMES})
        ctrl.reset()
        assert ctrl.current_positions() == {}


# ==========================================================================
# Command bridge
# ==========================================================================
class TestCommandFormat:
    def test_round_trip(self):
        positions = {"thumb": 300, "index": 450, "middle": 500, "ring": 480, "pinky": 460}
        frame, parsed = parse_command(format_command(positions, frame=7))
        assert frame == 7
        assert parsed == positions

    def test_missing_finger_written_as_hold(self):
        line = format_command({"index": 500}, frame=1)
        _, parsed = parse_command(line)
        assert parsed == {"index": 500}
        assert line.count("-1") == len(FINGER_ORDER) - 1

    def test_malformed_line_rejected(self):
        with pytest.raises(ValueError, match="Expected 6"):
            parse_command("1,2,3")

    def test_non_integer_rejected(self):
        with pytest.raises(ValueError, match="Non-integer"):
            parse_command("1,2,3,abc,5,6")


class TestAtomicWrite:
    def test_write_and_read(self, tmp_path):
        p = tmp_path / "cmd.txt"
        write_atomic(p, "hello")
        assert p.read_text() == "hello"

    def test_overwrite_leaves_no_temp_files(self, tmp_path):
        """A leaked temp file per frame would fill the disk in minutes."""
        p = tmp_path / "cmd.txt"
        for i in range(20):
            write_atomic(p, f"line {i}")
        assert p.read_text() == "line 19"
        leftovers = [f for f in tmp_path.iterdir() if f.name.startswith(".tmp_")]
        assert leftovers == [], f"temp files leaked: {leftovers}"

    def test_creates_parent_directory(self, tmp_path):
        p = tmp_path / "nested" / "deep" / "cmd.txt"
        write_atomic(p, "ok")
        assert p.read_text() == "ok"

    def test_file_is_never_partially_written(self, tmp_path):
        """Every observation of the file must be a complete, parseable command."""
        p = tmp_path / "cmd.txt"
        writer = CommandWriter(p)
        for i in range(50):
            writer.send({"index": 200 + i})
            frame, positions = read_command(p)   # read immediately after write
            assert frame == i + 1
            assert positions["index"] == 200 + i


class TestCommandWriter:
    def test_frame_numbers_increase_monotonically(self, tmp_path):
        writer = CommandWriter(tmp_path / "cmd.txt")
        frames = [writer.send({"index": 400}) for _ in range(5)]
        assert frames == [1, 2, 3, 4, 5]

    def test_reading_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_command(tmp_path / "nope.txt")

    def test_empty_file_rejected(self, tmp_path):
        p = tmp_path / "cmd.txt"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_command(p)


class TestEndToEnd:
    def test_open_hand_maps_to_open_positions(self):
        ctrl = HandController(deadband=0, smoothing=1.0)
        cmds = ctrl.update(all_flexions(open_hand()))
        for finger, pos in cmds.items():
            assert pos == DEFAULT_CALIBRATION[finger].motor_open

    def test_curled_hand_moves_toward_closed(self):
        ctrl = HandController(deadband=0, smoothing=1.0)
        open_cmds = ctrl.update(all_flexions(open_hand()))
        ctrl.reset()
        curl_cmds = ctrl.update(all_flexions(curled_hand()))
        for finger in curl_cmds:
            assert curl_cmds[finger] > open_cmds[finger], f"{finger} did not close"

    def test_full_pipeline_produces_valid_commands(self, tmp_path):
        ctrl = HandController()
        writer = CommandWriter(tmp_path / "cmd.txt")
        for lm in (open_hand(), curled_hand(), open_hand()):
            cmds = ctrl.update(all_flexions(lm))
            if cmds:
                writer.send(cmds)
                _, parsed = read_command(tmp_path / "cmd.txt")
                for pos in parsed.values():
                    assert POSITION_MIN <= pos <= POSITION_MAX
