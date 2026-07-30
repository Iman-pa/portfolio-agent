# ---------------------------------------------------------------------------
# rate_limit.py — simple file-backed rate limiting and usage logging.
#
# Streamlit Community Cloud runs a single process per app (no horizontal
# scaling on the free tier), so a local JSON file guarded by an in-process
# lock is sufficient here — no database or external cache needed. State is
# lost on redeploy/restart, which is an accepted tradeoff for a demo.
# ---------------------------------------------------------------------------

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Per-identifier (per-IP or per-session) cap within the rolling window.
MAX_PER_IDENTIFIER = 5

# Server-wide cap within the rolling window, across all visitors.
MAX_GLOBAL = 30

WINDOW_HOURS = 24

_LOGS_DIR = Path(__file__).parent.parent / "logs"
_STATE_PATH = _LOGS_DIR / "rate_limit_state.json"
_USAGE_LOG_PATH = _LOGS_DIR / "usage_log.jsonl"

# Guards all reads/writes to the state file — Streamlit runs each session's
# script in its own thread within the same process, so a plain Lock (not a
# multiprocess lock) is enough.
_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_state() -> dict:
    if not _STATE_PATH.exists():
        return {"identifiers": {}, "global": []}
    try:
        with open(_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable state file — reset rather than crash the app.
        return {"identifiers": {}, "global": []}


def _save_state(state: dict) -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename — avoids a torn/corrupt file if the
    # process is killed mid-write.
    tmp_path = _STATE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp_path, _STATE_PATH)


def _prune(timestamps: list, window_hours: int = WINDOW_HOURS) -> list:
    cutoff = _now() - timedelta(hours=window_hours)
    kept = []
    for ts in timestamps:
        try:
            if datetime.fromisoformat(ts) > cutoff:
                kept.append(ts)
        except ValueError:
            continue
    return kept


def try_consume(identifier: str) -> tuple[bool, str]:
    """Attempt to record one analysis run for `identifier`.

    Returns (allowed, message). If allowed is False, `message` is a friendly
    string safe to show directly to the visitor. Checking and recording
    happen under one lock so concurrent requests can't both slip through.
    """
    with _lock:
        state = _load_state()

        state["global"] = _prune(state["global"])
        identifiers = state.setdefault("identifiers", {})
        identifiers[identifier] = _prune(identifiers.get(identifier, []))

        if len(state["global"]) >= MAX_GLOBAL:
            return False, (
                "This demo has reached its shared daily usage cap. "
                "Please check back tomorrow — thanks for your patience."
            )

        if len(identifiers[identifier]) >= MAX_PER_IDENTIFIER:
            return False, (
                f"You've reached the limit of {MAX_PER_IDENTIFIER} analyses "
                "per 24 hours for this demo. Please check back later."
            )

        now_iso = _now().isoformat()
        state["global"].append(now_iso)
        identifiers[identifier].append(now_iso)
        _save_state(state)

    return True, ""


def log_usage(
    identifier: str,
    strategy: str,
    tickers: list[str],
    status: str = "success",
) -> None:
    """Append one line to the local usage log for later cost/usage review."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": _now().isoformat(),
        "identifier": identifier,
        "strategy": strategy,
        "tickers": tickers,
        "status": status,
    }
    with _lock:
        with open(_USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
