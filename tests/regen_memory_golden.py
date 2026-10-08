"""Write tests/fixtures/memory_golden/ from the harness as it stands: one member through three sessions against
the hashing fake model in test_memory_mode.py, with no --counselor_memory_mode flag. It was written at c85e278,
before the flag existed; regenerate it only for a deliberate change to what the harness writes.

    .venv/bin/python tests/regen_memory_golden.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_memory_mode import FILES, GOLDEN, produce  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        arc, _ = produce(Path(tmp), mp, None)
        if GOLDEN.exists():
            shutil.rmtree(GOLDEN)
        for name in FILES:
            (GOLDEN / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(arc / name, GOLDEN / name)
    print(f"wrote {len(FILES)} files to {GOLDEN}")


if __name__ == "__main__":
    main()
