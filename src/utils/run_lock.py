from __future__ import annotations

import atexit
import getpass
import json
import os
import socket
import sys
import time
from pathlib import Path

_LOCK_PATH: Path | None = None


def acquire_run_lock(output_dir: str, run_label: str) -> None:
    global _LOCK_PATH
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / ".active_run.lock"

    payload = {
        "pid": os.getpid(),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "run_label": run_label,
        "started_at_epoch": time.time(),
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": os.getcwd(),
    }

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            existing = json.loads(lock_path.read_text())
        except Exception:
            existing = {"raw": lock_path.read_text(errors="ignore") if lock_path.exists() else "<unreadable>"}
        print("\n[RUN LOCK] Another run is already active.", file=sys.stderr)
        print(f"[RUN LOCK] Lock file: {lock_path}", file=sys.stderr)
        print(f"[RUN LOCK] Details  : {existing}\n", file=sys.stderr)
        raise SystemExit(2)

    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)

    _LOCK_PATH = lock_path
    atexit.register(release_run_lock)
    print(f"[RUN LOCK] Acquired: {lock_path}")
    print(f"[RUN LOCK] Details : {payload}\n")


def release_run_lock() -> None:
    global _LOCK_PATH
    if _LOCK_PATH and _LOCK_PATH.exists():
        try:
            _LOCK_PATH.unlink()
            print(f"\n[RUN LOCK] Released: {_LOCK_PATH}")
        except Exception:
            pass
        finally:
            _LOCK_PATH = None