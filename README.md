# Vision-Controlled Prosthetic Hand

**Real-time gesture replication on a 3D-printed prosthetic hand, driven by a webcam alone — no gloves, no sensors on the user.**

MediaPipe hand landmarks → per-finger flexion angles → calibrated Dynamixel AX-12A motor positions on a Brunel Hand 2.0.

Built at the University of Reading (Medical Robotics & Prosthetics group project). **Selected for the department showcase and featured on the [UoR Biomedical Engineering Instagram](https://www.instagram.com/p/C47hZCxNJno/).**

Runs without a camera or a robot: `python -m pytest tests/` — 45 tests.

---

## Why angles rather than coordinates

The obvious approach is to map landmark positions straight to motor commands. It works at one distance from the camera and nowhere else, because landmark coordinates depend on how far away the hand is and where it sits in frame.

Joint **angles** are invariant to both. The angle at a knuckle is the same whether the user is close to the webcam or across the room, so a gesture produces the same grip either way. Flexion is computed from the dot product of the two bone segments meeting at each joint, summed across the PIP and DIP joints of each finger.

## The three problems between "angle computed" and "hand moves properly"

**Per-finger calibration.** No two people have the same range of motion, and no two fingers on one hand do either. Each finger is calibrated by recording its flexion fully open and fully closed, then normalised against that range. Without it, a user with limited mobility never reaches a full grip — the device simply doesn't respond to the range they actually have. There's a test for exactly that case.

**Mechanical limits.** The prosthetic's fingers can't travel the servo's full range. Commanding past the limit stalls the AX-12A, which then draws stall current and heats up. Every command is clamped to per-motor limits, and the clamp is tested against random out-of-range input.

**A deadband.** Landmark jitter of a degree or two produces a constant stream of tiny position updates. The motors chatter audibly and the gesture looks nervous rather than lifelike. Commands are re-sent only when the change exceeds a threshold, with EMA smoothing on top.

## The bug I fixed on revisiting this

The original build passed motor commands to MATLAB through a shared text file rewritten every frame. It worked — but it has a race condition: MATLAB can read the file midway through Python's write and get a truncated line, which parses to a garbage motor position and a jerking finger.

`src/bridge.py` writes to a temporary file in the same directory and then calls `os.replace`, which is atomic on POSIX and Windows. A reader sees either the complete old file or the complete new one, never a partial write. The temp file has to live on the same filesystem for the rename to be atomic, which is why it's written alongside the target rather than in `/tmp`.

There's a test that writes and reads 50 commands back to back and asserts every single observation is complete and parseable, plus one asserting no temp files leak — at 30 fps a leaked file per frame fills a disk quickly.

## Hardware

| Component | Detail |
|---|---|
| Hand | Brunel Hand 2.0, 3D-printed in-house |
| Actuation | 5 × Dynamixel AX-12A smart servos |
| Vision | Standard webcam, MediaPipe Hands (21 3-D landmarks) |
| Motor range | 0–1023 units over 300°, ≈0.29° per unit |
| Pipeline | Python (tracking, angles, mapping) → MATLAB (motor driving) |

## Run it

```bash
pip install -r requirements.txt
python demo.py            # synthetic hand poses — no camera or robot needed
python demo.py --live     # real webcam + MediaPipe, if installed
```

The synthetic path drives the full chain and verifies it: curl 0.0 → 1.0 maps
monotonically to motor positions, the deadband suppresses 10 near-identical
frames to 0 commands, a flexion of 1e6 degrees clamps inside the mechanical
limits, and the same pose at two distances from the camera gives an angle
difference of exactly 0.

## Use it

```python
from src.landmarks import all_flexions
from src.mapping import HandController
from src.bridge import CommandWriter

controller = HandController()          # loads default per-finger calibration
writer = CommandWriter("commands.txt")

# landmarks: (21, 3) array from MediaPipe
commands = controller.update(all_flexions(landmarks))
if commands:                            # only fingers that moved past the deadband
    writer.send(commands)
```

## Tests — 45

```bash
python -m pytest tests/ -v
```

Geometry is checked for the invariances it must have — translation and scale independence, coincident landmarks producing 180° rather than NaN, flexion never negative. Mapping is checked for endpoint accuracy, clamping under extreme input, inverted-motor handling, and that a reduced-range user still reaches full grip. The bridge is checked for atomicity, temp-file cleanup, and malformed-input rejection.

## Layout

```
src/landmarks.py   MediaPipe indices, joint angles, per-finger flexion
src/mapping.py     calibration, flexion → AX-12A position, deadband + smoothing
src/bridge.py      atomic command file for the MATLAB side
tests/test_hand.py 45 tests
```

## My contribution to the group project

Co-developed the Python tracking pipeline (OpenCV + MediaPipe) extracting finger joint positions and flexion angles; built the Python↔MATLAB data transfer; wrote and calibrated the MATLAB scripts converting flexion to motor positions within each motor's limits; assembled and tested the hardware; and ran the live demonstration, including tuning motor response for gestures like grasping and pointing.

## What I'd do next

- Replace the file bridge with a socket or shared memory — atomic writes fix correctness, not the latency of hitting the disk every frame
- Guided calibration routine that walks a user through open/closed per finger, rather than hard-coded defaults
- Grip force feedback so the hand stops closing on contact instead of on commanded position
- Drive the Dynamixels from Python directly and drop MATLAB from the loop

## Licence

MIT — see [LICENSE](LICENSE).

---

Sam Tsang · [sammytsang.github.io](https://sammytsang.github.io) · [LinkedIn](https://www.linkedin.com/in/sam-tsang-65608529a)
