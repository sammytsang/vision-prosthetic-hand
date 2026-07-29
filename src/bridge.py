"""Python -> MATLAB command bridge.

The original build passed motor commands to MATLAB through a shared text file
that Python rewrote every frame. It worked, but it has a race condition: MATLAB
can read the file midway through Python's write and get a truncated line, which
parses to a garbage motor position.

`write_atomic` fixes that. It writes to a temporary file in the same directory
and then calls `os.replace`, which is atomic on POSIX and Windows — a reader
sees either the complete old file or the complete new one, never a partial
write. The temp file must be on the same filesystem, hence writing it alongside
the target rather than in /tmp.

`FrameCounter` adds a monotonically increasing frame number so the reader can
tell a fresh command from a stale one when the writer stalls.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def format_command(positions: dict[str, int], frame: int = 0) -> str:
    """Serialise motor positions to a single line.

    Format: `frame,thumb,index,middle,ring,pinky`
    Missing fingers are written as -1, which the reader treats as "hold".
    """
    values = [str(positions.get(name, -1)) for name in FINGER_ORDER]
    return f"{frame}," + ",".join(values)


def parse_command(line: str) -> tuple[int, dict[str, int]]:
    """Inverse of `format_command`. Raises on malformed input."""
    parts = line.strip().split(",")
    if len(parts) != len(FINGER_ORDER) + 1:
        raise ValueError(
            f"Expected {len(FINGER_ORDER) + 1} comma-separated fields, got {len(parts)}"
        )
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Non-integer field in command line: {line!r}") from exc

    frame = values[0]
    positions = {
        name: v for name, v in zip(FINGER_ORDER, values[1:]) if v >= 0
    }
    return frame, positions


def write_atomic(path: str | Path, content: str) -> None:
    """Write `content` to `path` atomically.

    A reader polling the file will never observe a partial write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())   # ensure it's on disk before the rename
        os.replace(tmp, path)       # atomic
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class CommandWriter:
    """Writes motor commands for MATLAB to consume, with frame numbering."""

    path: Path
    frame: int = field(default=0, init=False)

    def send(self, positions: dict[str, int]) -> int:
        """Write one command line. Returns the frame number used."""
        self.frame += 1
        write_atomic(self.path, format_command(positions, self.frame))
        return self.frame


def read_command(path: str | Path) -> tuple[int, dict[str, int]]:
    """Read the most recent command. Raises if the file is missing or malformed."""
    text = Path(path).read_text().strip()
    if not text:
        raise ValueError("Command file is empty")
    return parse_command(text.splitlines()[-1])
