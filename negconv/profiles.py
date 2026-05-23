"""Named parameter profiles for film stock + scanner combinations."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .params import NegconvParams

PROFILE_DIR = Path.home() / ".negconv" / "profiles"


def _sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-.]', '_', name)


def ensure_profile_dir() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def save_profile(name: str, params: NegconvParams, notes: str = "") -> Path:
    """Save current params as a named profile. Returns the profile path."""
    ensure_profile_dir()
    data = {
        "name": name,
        "created": datetime.now().isoformat(),
        "notes": notes,
        "params": {
            "dmin": params.dmin.tolist(),
            "d_max": params.d_max,
            "wb_high": params.wb_high.tolist(),
            "wb_low": params.wb_low.tolist(),
            "offset": params.offset,
            "exposure": params.exposure,
            "black": params.black,
            "gamma": params.gamma,
            "soft_clip": params.soft_clip,
        },
    }
    path = PROFILE_DIR / f"{_sanitize_filename(name)}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def load_profile(name: str) -> NegconvParams:
    """Load a named profile and return NegconvParams."""
    path = PROFILE_DIR / f"{_sanitize_filename(name)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Profile not found: {name}")
    with open(path) as f:
        data = json.load(f)
    p = data["params"]
    return NegconvParams(
        dmin=np.array(p["dmin"], dtype=np.float32),
        d_max=p["d_max"],
        wb_high=np.array(p["wb_high"], dtype=np.float32),
        wb_low=np.array(p["wb_low"], dtype=np.float32),
        offset=p["offset"],
        exposure=p["exposure"],
        black=p["black"],
        gamma=p["gamma"],
        soft_clip=p["soft_clip"],
    )


def list_profiles() -> list[dict]:
    """Return [{name, created, notes, path}] for all saved profiles."""
    ensure_profile_dir()
    profiles = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            profiles.append({
                "name": data.get("name", path.stem),
                "created": data.get("created", ""),
                "notes": data.get("notes", ""),
                "path": str(path),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return profiles


def delete_profile(name: str) -> bool:
    """Delete a named profile. Returns True if deleted."""
    path = PROFILE_DIR / f"{_sanitize_filename(name)}.json"
    if path.is_file():
        path.unlink()
        return True
    return False
