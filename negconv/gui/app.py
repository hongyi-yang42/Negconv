from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from ..io import apply_orientation, extract_exif, is_raw, read_image, write_image, write_jpeg, write_heic, RAW_EXTENSIONS
from ..params import NegconvParams, auto_detect, save_params, load_params, PARAM_CATEGORIES, carry_fields_for_categories
from ..pipeline import invert
from ..color import srgb_to_rec2020, rec2020_to_srgb, recover_highlights
from ..profiles import save_profile, load_profile, list_profiles, delete_profile
from ..lut import parse_cube, apply_lut
from .viewer import make_preview

PREVIEW_MAX_WIDTH = 1200
RECENT_FILE = Path.home() / ".negconv" / "recent.json"
SETTINGS_FILE = Path.home() / ".negconv" / "settings.json"
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | {'.tif', '.tiff'}

THUMB_DIR = Path.home() / ".negconv" / "thumbs"
THUMB_WIDTH = 120
THUMB_WINDOW = 10  # generate ±10 around current index
_thumb_executor = ThreadPoolExecutor(max_workers=2)
_thumb_queued: set[int] = set()  # indices already queued for generation

DEFAULT_SETTINGS = {
    "carry_categories": {"tone": True, "wb": True, "film_base": False, "geometry": True},
    "auto_redetect_on_crop": True,
    "preview_quality": 90,
    "preview_max_width": 1200,
    "highlight_recovery": False,
    "include_subdirs": False,
}


def _load_settings() -> dict:
    if SETTINGS_FILE.is_file():
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
        merged = {**DEFAULT_SETTINGS, **saved}
        return merged
    return dict(DEFAULT_SETTINGS)


def _save_settings(settings: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


@dataclass
class ParamHistory:
    """In-memory undo/redo stack for params. Max 50 entries.

    Stack stores snapshots AFTER each action. Index points to the current state.
    undo() decrements index, redo() increments it.
    """
    stack: list[dict] = field(default_factory=list)
    index: int = -1  # points to current state in stack
    max_depth: int = 50

    def push(self, snapshot: dict) -> None:
        """Push a new state (after an action). Discards redo entries ahead."""
        # Discard any redo states ahead of current index
        self.stack = self.stack[:self.index + 1]
        self.stack.append(snapshot)
        if len(self.stack) > self.max_depth:
            self.stack.pop(0)
        self.index = len(self.stack) - 1

    def undo(self) -> dict | None:
        """Go back one step. Returns the previous state, or None if at start."""
        if self.index <= 0:
            return None
        self.index -= 1
        return copy.deepcopy(self.stack[self.index])

    def redo(self) -> dict | None:
        """Go forward one step. Returns the next state, or None if at end."""
        if self.index >= len(self.stack) - 1:
            return None
        self.index += 1
        return copy.deepcopy(self.stack[self.index])

    def clear(self) -> None:
        self.stack.clear()
        self.index = -1


def _snapshot_params(state: GuiState) -> dict:
    """Capture a serializable snapshot of state params."""
    return {
        "dmin": state.params.dmin.tolist(),
        "d_max": state.params.d_max,
        "wb_high": state.params.wb_high.tolist(),
        "wb_low": state.params.wb_low.tolist(),
        "offset": state.params.offset,
        "exposure": state.params.exposure,
        "black": state.params.black,
        "gamma": state.params.gamma,
        "soft_clip": state.params.soft_clip,
    }


def _restore_snapshot(state: GuiState, snap: dict) -> None:
    """Apply a snapshot back to state params."""
    p = state.params
    p.dmin = np.array(snap["dmin"], dtype=np.float32)
    p.d_max = snap["d_max"]
    p.wb_high = np.array(snap["wb_high"], dtype=np.float32)
    p.wb_low = np.array(snap["wb_low"], dtype=np.float32)
    p.offset = snap["offset"]
    p.exposure = snap["exposure"]
    p.black = snap["black"]
    p.gamma = snap["gamma"]
    p.soft_clip = snap["soft_clip"]


@dataclass
class GuiState:
    original_img: np.ndarray | None = None
    params: NegconvParams = field(default_factory=NegconvParams.color_film)
    file_path: str = ""
    orig_preview: bytes = b""
    result_preview: bytes = b""
    preview_dims: tuple[int, int] = (0, 0)  # (width, height) of preview image
    crop_rect: dict | None = None  # {"x","y","w","h"} in original coords
    source_exif: bytes | None = None
    orientation: int = 0      # 0=normal, 1=90°CW, 2=180°, 3=270°CW
    flip_h: bool = False
    flip_v: bool = False
    is_raw_input: bool = False  # True if source is RAW (already Rec.2020)
    black_level: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    directory_files: list[str] = field(default_factory=list)
    current_index: int = 0
    carry_categories: dict = field(default_factory=lambda: {"tone": True, "wb": True, "film_base": False, "geometry": True})
    settings: dict = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    history: ParamHistory = field(default_factory=ParamHistory)
    lut_path: str = ""
    lut_data: dict | None = None
    highlight_recovery: bool = False
    result_rec2020: np.ndarray | None = None


def _sample_patch(img: np.ndarray, orig_x: int, orig_y: int, patch: int = 5) -> np.ndarray:
    """Sample a patch around (orig_x, orig_y), return per-channel median."""
    h, w = img.shape[:2]
    half = patch // 2
    y0 = max(orig_y - half, 0)
    y1 = min(orig_y + half + 1, h)
    x0 = max(orig_x - half, 0)
    x1 = min(orig_x + half + 1, w)
    region = img[y0:y1, x0:x1, :]
    return np.median(region.reshape(-1, 3), axis=0).astype(np.float32)


def _get_pipeline_input(state: GuiState) -> np.ndarray:
    """Return the image region to run through the pipeline (cropped or full)."""
    if state.crop_rect and state.original_img is not None:
        r = state.crop_rect
        return state.original_img[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]]
    return state.original_img


_PIL_ROTATE = {1: 4, 2: 3, 3: 2}  # PIL: 4=ROTATE_270(=90°CW), 3=ROTATE_180, 2=ROTATE_90(=270°CW)


def _orient_preview(jpeg_bytes: bytes, state: GuiState) -> bytes:
    """Apply rotation then flip to preview JPEG bytes. Flip is relative to displayed orientation."""
    if state.orientation == 0 and not state.flip_h and not state.flip_v:
        return jpeg_bytes
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(jpeg_bytes))
    if state.orientation in _PIL_ROTATE:
        img = img.transpose(_PIL_ROTATE[state.orientation])
    if state.flip_h:
        img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
    if state.flip_v:
        img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _sidecar_path(file_path: str) -> str:
    """Return the sidecar path for a given source file."""
    return str(file_path) + ".negconv.json"


def _auto_save(state: GuiState) -> bool:
    """Save params + crop_rect to sidecar. Returns True if saved."""
    if not state.file_path:
        return False
    sp = _sidecar_path(state.file_path)
    data = {
        "dmin": state.params.dmin.tolist(),
        "d_max": state.params.d_max,
        "wb_high": state.params.wb_high.tolist(),
        "wb_low": state.params.wb_low.tolist(),
        "offset": state.params.offset,
        "exposure": state.params.exposure,
        "black": state.params.black,
        "gamma": state.params.gamma,
        "soft_clip": state.params.soft_clip,
    }
    if state.crop_rect is not None:
        data["crop_rect"] = state.crop_rect
    data["orientation"] = state.orientation
    data["flip_h"] = state.flip_h
    data["flip_v"] = state.flip_v
    Path(sp).parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(data, f, indent=2)
    return True


def _load_sidecar(file_path: str) -> dict | None:
    """Load sidecar JSON if it exists. Returns dict or None."""
    sp = _sidecar_path(file_path)
    if os.path.isfile(sp):
        with open(sp) as f:
            return json.load(f)
    return None


def _apply_sidecar(state: GuiState, data: dict) -> None:
    """Apply sidecar data to state params and crop_rect."""
    p = state.params
    if "dmin" in data:
        p.dmin = np.array(data["dmin"], dtype=np.float32)
    for key in ("d_max", "exposure", "black", "gamma", "soft_clip", "offset"):
        if key in data:
            setattr(p, key, float(data[key]))
    for key in ("wb_high", "wb_low"):
        if key in data:
            setattr(p, key, np.array(data[key], dtype=np.float32))
    state.crop_rect = data.get("crop_rect", None)
    state.orientation = data.get("orientation", 0)
    state.flip_h = data.get("flip_h", False)
    state.flip_v = data.get("flip_v", False)


def _load_recent() -> list[dict]:
    """Load recent files list from disk."""
    if RECENT_FILE.is_file():
        with open(RECENT_FILE) as f:
            return json.load(f)
    return []


def _save_recent(recent: list[dict]) -> None:
    """Save recent files list to disk."""
    RECENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RECENT_FILE, "w") as f:
        json.dump(recent, f, indent=2)


def _add_recent(file_path: str) -> None:
    """Add a file to the recent list (newest first, max 10)."""
    recent = _load_recent()
    # Remove duplicates
    recent = [r for r in recent if r["path"] != file_path]
    recent.insert(0, {"path": file_path, "timestamp": datetime.now(timezone.utc).isoformat()})
    _save_recent(recent[:10])


def _scan_directory(file_path: str) -> list[str]:
    """Scan parent directory for supported image files, sorted alphabetically."""
    return _scan_directory_simple(Path(file_path).parent)


def _scan_directory_simple(dir_path: str | Path, include_subdirs: bool = False) -> list[str]:
    """Scan a directory for supported image files, sorted alphabetically."""
    parent = Path(dir_path)
    if not parent.is_dir():
        return []
    files = []
    if include_subdirs:
        for root, _dirs, fnames in os.walk(parent):
            for fname in sorted(fnames):
                if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(str(Path(root) / fname))
        files.sort()
    else:
        for f in sorted(parent.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file():
                files.append(str(f))
    return files


def _thumb_path(file_path: str) -> Path:
    mtime = os.path.getmtime(file_path)
    key = f"{file_path}:{mtime}"
    h = hashlib.md5(key.encode()).hexdigest()
    return THUMB_DIR / f"{h}.jpg"


def _generate_thumb(file_path: str) -> None:
    tp = _thumb_path(file_path)
    if tp.exists():
        return
    tp.parent.mkdir(parents=True, exist_ok=True)
    ext = Path(file_path).suffix.lower()
    try:
        if ext in {'.arw', '.cr2', '.cr3', '.nef', '.raf', '.dng'}:
            import rawpy
            raw = rawpy.imread(file_path)
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                from PIL import Image as PILImage
                img = PILImage.open(io.BytesIO(thumb.data))
            else:
                from PIL import Image as PILImage
                img = PILImage.fromarray(thumb.data)
        else:
            from PIL import Image as PILImage
            img = PILImage.open(file_path)
        ratio = THUMB_WIDTH / img.width
        img = img.resize((THUMB_WIDTH, int(img.height * ratio)), PILImage.LANCZOS)
        img = img.convert("RGB")
        img.save(str(tp), "JPEG", quality=70)
    except Exception:
        pass


def _start_thumbnails(files: list[str]) -> None:
    """DEPRECATED — use _ensure_thumb_window() instead."""
    pass


def _ensure_thumb_window(center: int, state: GuiState) -> None:
    """Generate thumbnails for indices [center-WINDOW, center+WINDOW]."""
    files = state.directory_files
    lo = max(0, center - THUMB_WINDOW)
    hi = min(len(files), center + THUMB_WINDOW + 1)
    for i in range(lo, hi):
        if i not in _thumb_queued:
            tp = _thumb_path(files[i])
            if not tp.exists():
                _thumb_queued.add(i)
                _thumb_executor.submit(_generate_thumb, files[i])


def _load_file(state: GuiState, path: str) -> bool:
    """Load a file into state. Returns True if sidecar was loaded.

    RAW files arrive as linear Rec.2020 (rawpy handles camera matrix).
    TIFF files arrive as linear sRGB and are converted to Rec.2020.
    """
    img, raw_input = read_image(path)
    state.is_raw_input = raw_input
    if not raw_input:
        img = srgb_to_rec2020(img)
    state.original_img = img
    state.file_path = path

    # Read black level per channel for RAW files
    if raw_input:
        try:
            import rawpy
            with rawpy.imread(path) as raw:
                bl = raw.black_level_per_channel
                state.black_level = [int(v) for v in bl] if bl else [0, 0, 0, 0]
        except Exception:
            state.black_level = [0, 0, 0, 0]
    else:
        state.black_level = [0, 0, 0, 0]

    state.params = auto_detect(img)
    state.crop_rect = None
    state.source_exif = extract_exif(path)
    state.orientation = 0
    state.flip_h = False
    state.flip_v = False
    state.history.clear()

    sidecar_loaded = False
    sidecar = _load_sidecar(path)
    if sidecar is not None:
        _apply_sidecar(state, sidecar)
        sidecar_loaded = True

    max_w = state.settings.get("preview_max_width", PREVIEW_MAX_WIDTH)
    state.orig_preview = make_preview(img, max_w)
    from PIL import Image as PILImage
    pil_preview = PILImage.open(io.BytesIO(state.orig_preview))
    state.preview_dims = (pil_preview.width, pil_preview.height)

    # Push initial state into undo history
    state.history.push(_snapshot_params(state))

    return sidecar_loaded


def _run_pipeline(state: GuiState) -> None:
    """Run inversion pipeline and store result preview.

    Pipeline runs in Rec.2020 working space. Result is converted back to
    sRGB for preview and export (except TIFF-32f which stays in Rec.2020).

    When a LUT is loaded, gamma/soft_clip are set to identity so stages 4+5
    become no-ops, then the LUT is applied to the Stage 3 output.
    """
    pipeline_input = _get_pipeline_input(state)

    if state.highlight_recovery:
        pipeline_input = recover_highlights(pipeline_input)

    if state.lut_data is not None:
        saved_gamma = state.params.gamma
        saved_soft_clip = state.params.soft_clip
        state.params.gamma = 1.0
        state.params.soft_clip = 1.0
        result = invert(pipeline_input, state.params)
        state.params.gamma = saved_gamma
        state.params.soft_clip = saved_soft_clip
        result = np.clip(result, 0.0, 1.0)
        result = apply_lut(result, state.lut_data)
    else:
        result = invert(pipeline_input, state.params)

    state.result_rec2020 = result
    result_srgb = rec2020_to_srgb(result)
    result_srgb = np.clip(result_srgb, 0, None)
    max_w = state.settings.get("preview_max_width", PREVIEW_MAX_WIDTH)
    quality = state.settings.get("preview_quality", 90)
    state.result_preview = make_preview(result_srgb, max_w, quality=quality)


def _snapshot_carry(state: GuiState) -> dict:
    """Snapshot all carryable fields from current state."""
    return {
        "params": {
            "dmin": state.params.dmin.copy(),
            "d_max": state.params.d_max,
            "wb_high": state.params.wb_high.copy(),
            "wb_low": state.params.wb_low.copy(),
            "offset": state.params.offset,
            "exposure": state.params.exposure,
            "black": state.params.black,
            "gamma": state.params.gamma,
            "soft_clip": state.params.soft_clip,
        },
        "crop_rect": state.crop_rect,
        "orientation": state.orientation,
        "flip_h": state.flip_h,
        "flip_v": state.flip_v,
    }


def _apply_carry(state: GuiState, snapshot: dict, categories: dict) -> None:
    """Apply carried fields from snapshot to state, using enabled categories."""
    enabled_fields = carry_fields_for_categories(categories)
    geo_fields = {"crop_rect", "orientation", "flip_h", "flip_v"}

    sp = snapshot["params"]
    for fname in enabled_fields:
        if fname in geo_fields:
            continue
        if fname in sp:
            val = sp[fname]
            setattr(state.params, fname, val.copy() if isinstance(val, np.ndarray) else val)

    if "crop_rect" in enabled_fields:
        state.crop_rect = snapshot["crop_rect"]
    if "orientation" in enabled_fields:
        state.orientation = snapshot["orientation"]
    if "flip_h" in enabled_fields:
        state.flip_h = snapshot["flip_h"]
    if "flip_v" in enabled_fields:
        state.flip_v = snapshot["flip_v"]


def _redetect_in_crop(state: GuiState) -> None:
    """Re-detect Dmin/Dmax within crop + 5% inset, update params, run pipeline."""
    img = state.original_img

    if state.crop_rect:
        r = state.crop_rect
        img = img[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]]

    h, w = img.shape[:2]
    mx = max(1, int(w * 0.05))
    my = max(1, int(h * 0.05))
    img_inner = img[my:h - my, mx:w - mx]

    new_params = auto_detect(img_inner)
    state.params.dmin = new_params.dmin
    state.params.d_max = new_params.d_max


def create_app() -> Flask:
    app = Flask(__name__)
    state = GuiState(settings=_load_settings())
    state.highlight_recovery = state.settings.get("highlight_recovery", False)
    state.carry_categories = state.settings.get("carry_categories", DEFAULT_SETTINGS["carry_categories"])

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/load", methods=["POST"])
    def api_load():
        data = request.get_json(force=True)
        path = data.get("path", "")
        if not path or not os.path.isfile(path):
            return jsonify({"error": f"File not found: {path}"}), 400

        try:
            sidecar_loaded = _load_file(state, path)
        except Exception as e:
            return jsonify({"error": f"Failed to read: {e}"}), 400

        state.directory_files = _scan_directory(path)
        try:
            state.current_index = state.directory_files.index(path)
        except ValueError:
            state.current_index = 0

        _add_recent(path)
        _thumb_queued.clear()
        _ensure_thumb_window(state.current_index, state)

        h, w = state.original_img.shape[:2]
        result = {
            "preview": "/api/preview/orig",
            "params": _params_to_dict(state.params, state.crop_rect),
            "crop_rect": state.crop_rect,
            "filename": Path(path).name,
            "dims": [h, w],
            "preview_dims": list(state.preview_dims),
            "sidecar_loaded": sidecar_loaded,
            "orientation": state.orientation,
            "flip_h": state.flip_h,
            "flip_v": state.flip_v,
            "current_index": state.current_index,
            "total_files": len(state.directory_files),
        }
        return jsonify(result)

    @app.route("/api/preview/orig")
    def api_preview_orig():
        if not state.orig_preview:
            return "", 404
        data = _orient_preview(state.orig_preview, state)
        return send_file(io.BytesIO(data), mimetype="image/jpeg")

    @app.route("/api/preview/result")
    def api_preview_result():
        if not state.result_preview:
            return "", 404
        data = _orient_preview(state.result_preview, state)
        return send_file(io.BytesIO(data), mimetype="image/jpeg")

    @app.route("/api/invert", methods=["POST"])
    def api_invert():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400

        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        return jsonify({
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
        })

    @app.route("/api/pick-dmin", methods=["POST"])
    def api_pick_dmin():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400

        data = request.get_json(force=True)
        orig_x, orig_y = data.get("x", 0), data.get("y", 0)

        # JS already maps preview→original coords via previewToOriginalCoords
        dmin = _sample_patch(state.original_img, orig_x, orig_y)
        state.params.dmin = dmin

        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        state.history.push(_snapshot_params(state))
        return jsonify({
            "dmin": dmin.tolist(),
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
            "auto_saved": _auto_save(state),
        })

    @app.route("/api/params", methods=["GET"])
    def api_get_params():
        return jsonify(_params_to_dict(state.params, state.crop_rect))

    @app.route("/api/params", methods=["POST"])
    def api_set_params():
        data = request.get_json(force=True)
        _update_params_from_dict(state.params, data)
        state.history.push(_snapshot_params(state))
        saved = _auto_save(state)
        return jsonify({**_params_to_dict(state.params, state.crop_rect), "auto_saved": saved})

    @app.route("/api/preset/<name>", methods=["POST"])
    def api_preset(name):
        if name == "color":
            state.params = NegconvParams.color_film()
        elif name == "bw":
            state.params = NegconvParams.bw_film()
        else:
            return jsonify({"error": f"Unknown preset: {name}"}), 400
        state.history.push(_snapshot_params(state))
        return jsonify(_params_to_dict(state.params, state.crop_rect))

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        """Reset current image to defaults: delete sidecar, reload fresh."""
        if not state.file_path:
            return jsonify({"error": "No file loaded"}), 400
        # Delete sidecar
        sp = _sidecar_path(state.file_path)
        if os.path.isfile(sp):
            os.unlink(sp)
        # Reload with fresh defaults
        try:
            _load_file(state, state.file_path)
        except Exception as e:
            return jsonify({"error": f"Failed to reload: {e}"}), 400
        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500
        h, w = state.original_img.shape[:2]
        return jsonify({
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
            "crop_rect": state.crop_rect,
            "filename": Path(state.file_path).name,
            "dims": [h, w],
            "preview_dims": list(state.preview_dims),
            "sidecar_loaded": False,
            "orientation": state.orientation,
            "flip_h": state.flip_h,
            "flip_v": state.flip_v,
        })

    # ---- Profile endpoints ----
    @app.route("/api/profiles", methods=["GET"])
    def api_list_profiles():
        return jsonify(list_profiles())

    @app.route("/api/profiles", methods=["POST"])
    def api_save_profile():
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "Profile name required"}), 400
        path = save_profile(name, state.params)
        return jsonify({"ok": True, "name": name, "path": str(path)})

    @app.route("/api/profiles/load", methods=["POST"])
    def api_load_profile():
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "Profile name required"}), 400
        try:
            loaded = load_profile(name)
        except FileNotFoundError:
            return jsonify({"error": f"Profile not found: {name}"}), 404
        # Copy loaded params into state
        state.params.dmin = loaded.dmin.copy()
        state.params.d_max = loaded.d_max
        state.params.wb_high = loaded.wb_high.copy()
        state.params.wb_low = loaded.wb_low.copy()
        state.params.offset = loaded.offset
        state.params.exposure = loaded.exposure
        state.params.black = loaded.black
        state.params.gamma = loaded.gamma
        state.params.soft_clip = loaded.soft_clip
        _auto_save(state)
        return jsonify(_params_to_dict(state.params, state.crop_rect))

    @app.route("/api/profiles/<name>", methods=["DELETE"])
    def api_delete_profile(name):
        deleted = delete_profile(name)
        if not deleted:
            return jsonify({"error": f"Profile not found: {name}"}), 404
        return jsonify({"ok": True})

    # ---- Undo / Redo ----
    @app.route("/api/undo", methods=["POST"])
    def api_undo():
        snap = state.history.undo()
        if snap is None:
            return jsonify({"error": "Nothing to undo"}), 400
        _restore_snapshot(state, snap)
        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500
        _auto_save(state)
        return jsonify({
            "params": _params_to_dict(state.params, state.crop_rect),
            "preview": "/api/preview/result",
        })

    @app.route("/api/redo", methods=["POST"])
    def api_redo():
        snap = state.history.redo()
        if snap is None:
            return jsonify({"error": "Nothing to redo"}), 400
        _restore_snapshot(state, snap)
        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500
        _auto_save(state)
        return jsonify({
            "params": _params_to_dict(state.params, state.crop_rect),
            "preview": "/api/preview/result",
        })

    # ---- WB eyedropper ----
    @app.route("/api/pick-wb", methods=["POST"])
    def api_pick_wb():
        """Sample a neutral point to set WB via log-density correction.

        Samples from state.original_img at the clicked coords, computes
        per-channel log density relative to Dmin, then adjusts wb_high
        to equalize densities (green anchor).
        """
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400

        data = request.get_json(force=True)
        orig_x, orig_y = data.get("x", 0), data.get("y", 0)

        patch = _sample_patch(state.original_img, orig_x, orig_y)
        dmin = state.params.dmin

        safe_patch = np.maximum(patch, np.float32(1e-6))
        safe_dmin = np.maximum(dmin, np.float32(1e-6))
        log_density = np.log10(safe_patch / safe_dmin)

        if np.any(np.abs(log_density) < 0.01):
            return jsonify({"error": "Too close to film base — pick an exposed area"}), 400

        green_ld = log_density[1]
        green_wb = state.params.wb_high[1]
        new_wb = green_wb * green_ld / log_density
        state.params.wb_high = np.clip(new_wb, 0.5, 2.0).astype(np.float32)

        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        state.history.push(_snapshot_params(state))
        return jsonify({
            "wb_high": state.params.wb_high.tolist(),
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
            "auto_saved": _auto_save(state),
        })

    @app.route("/api/info")
    def api_info():
        h, w = state.original_img.shape[:2] if state.original_img is not None else (0, 0)
        return jsonify({
            "filename": Path(state.file_path).name if state.file_path else "",
            "dims": [h, w],
            "is_raw": state.is_raw_input,
            "black_level": state.black_level,
            "orientation": state.orientation,
            "flip_h": state.flip_h,
            "flip_v": state.flip_v,
        })

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        state.original_img = None
        state.params = NegconvParams.color_film()
        state.file_path = ""
        state.orig_preview = b""
        state.result_preview = b""
        state.preview_dims = (0, 0)
        state.crop_rect = None
        state.directory_files = []
        state.current_index = 0
        state.source_exif = None
        state.orientation = 0
        state.flip_h = False
        state.flip_v = False
        state.is_raw_input = False
        state.black_level = [0, 0, 0, 0]
        state.history.clear()
        return jsonify({"ok": True})

    @app.route("/api/crop", methods=["POST"])
    def api_set_crop():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400

        data = request.get_json(force=True)
        state.crop_rect = {
            "x": int(data["x"]), "y": int(data["y"]),
            "w": int(data["w"]), "h": int(data["h"]),
        }

        # Auto re-detect Dmin/Dmax within crop if setting enabled
        if state.settings.get("auto_redetect_on_crop", True):
            _redetect_in_crop(state)

        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        return jsonify({
            "crop_rect": state.crop_rect,
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
            "auto_saved": _auto_save(state),
        })

    @app.route("/api/crop", methods=["DELETE"])
    def api_clear_crop():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400

        state.crop_rect = None

        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        return jsonify({
            "crop_rect": None,
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
            "auto_saved": _auto_save(state),
        })

    @app.route("/api/rotate", methods=["POST"])
    def api_rotate():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400
        data = request.get_json(force=True)
        action = data.get("action", "cw")
        if action == "cw":
            state.orientation = (state.orientation + 1) % 4
        elif action == "ccw":
            state.orientation = (state.orientation + 3) % 4
        elif action == "180":
            state.orientation = (state.orientation + 2) % 4
        preview_url = "/api/preview/result" if state.result_preview else "/api/preview/orig"
        return jsonify({
            "orientation": state.orientation,
            "flip_h": state.flip_h,
            "flip_v": state.flip_v,
            "preview": preview_url,
            "auto_saved": _auto_save(state),
        })

    @app.route("/api/flip", methods=["POST"])
    def api_flip():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400
        data = request.get_json(force=True)
        axis = data.get("axis", "h")
        if axis == "h":
            state.flip_h = not state.flip_h
        elif axis == "v":
            state.flip_v = not state.flip_v
        preview_url = "/api/preview/result" if state.result_preview else "/api/preview/orig"
        return jsonify({
            "orientation": state.orientation,
            "flip_h": state.flip_h,
            "flip_v": state.flip_v,
            "preview": preview_url,
            "auto_saved": _auto_save(state),
        })

    @app.route("/api/re-detect", methods=["POST"])
    def api_redetect():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400

        _redetect_in_crop(state)

        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        saved = _auto_save(state)
        return jsonify({
            "dmin": state.params.dmin.tolist(),
            "d_max": state.params.d_max,
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
            "auto_saved": saved,
        })

    @app.route("/api/histogram")
    def api_histogram():
        source = request.args.get("source", "preview")
        if source == "precise":
            if state.result_rec2020 is None:
                return jsonify({"error": "No result"}), 404
            from ..color import rec2020_to_srgb
            arr = np.clip(rec2020_to_srgb(state.result_rec2020), 0, 1)
            hist = {}
            for i, ch in enumerate(("r", "g", "b")):
                counts, _ = np.histogram(arr[:, :, i], bins=256, range=(0, 1))
                hist[ch] = counts.tolist()
            return jsonify(hist)
        # Default: from preview JPEG (fast, uint8)
        if not state.result_preview:
            return jsonify({"error": "No result"}), 404
        from PIL import Image as PILImage
        import io as _io
        pil = PILImage.open(_io.BytesIO(state.result_preview))
        arr = np.array(pil)
        hist = {}
        for i, ch in enumerate(("r", "g", "b")):
            counts, _ = np.histogram(arr[:, :, i], bins=256, range=(0, 256))
            hist[ch] = counts.tolist()
        return jsonify(hist)

    @app.route("/api/auto-save", methods=["POST"])
    def api_auto_save():
        saved = _auto_save(state)
        return jsonify({"auto_saved": saved})

    @app.route("/api/recent")
    def api_recent():
        recent = _load_recent()
        def _is_temp_path(p: str) -> bool:
            import re
            return bool(re.match(r'^(/private)?/tmp/tmp', p) or re.match(r'^(/private)?/var/folders/', p))
        recent = [r for r in recent
                  if os.path.isfile(r.get("path", ""))
                  and not _is_temp_path(r["path"])]
        return jsonify(recent)

    @app.route("/api/directory")
    def api_directory():
        files = []
        for i, path in enumerate(state.directory_files):
            files.append({
                "path": path,
                "name": Path(path).name,
                "index": i,
                "has_sidecar": os.path.isfile(_sidecar_path(path)),
            })
        return jsonify({
            "files": files,
            "current_index": state.current_index,
            "carry_categories": state.carry_categories,
        })

    @app.route("/api/navigate", methods=["POST"])
    def api_navigate():
        data = request.get_json(force=True)

        if not state.directory_files:
            return jsonify({"error": "No directory loaded"}), 400

        if "index" in data:
            target = data["index"]
        elif "direction" in data:
            d = data["direction"]
            target = state.current_index + (1 if d == "next" else -1 if d == "prev" else 0)
        else:
            return jsonify({"error": "Provide index or direction"}), 400

        if target < 0 or target >= len(state.directory_files):
            return jsonify({"error": "No more files"}), 400

        # 1. Snapshot current state for carry
        carry_snapshot = _snapshot_carry(state)

        # 2. Auto-save current file
        _auto_save(state)

        new_path = state.directory_files[target]
        try:
            sidecar_loaded = _load_file(state, new_path)
        except Exception as e:
            return jsonify({"error": f"Failed to read: {e}"}), 400
        state.current_index = target
        _ensure_thumb_window(target, state)

        # 4. If no sidecar, apply carry using enabled categories
        if not sidecar_loaded:
            _apply_carry(state, carry_snapshot, state.carry_categories)
            if state.carry_categories.get("geometry", False):
                _redetect_in_crop(state)

        # 5. Always run pipeline so result preview is available
        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        # 6. Add to recent
        _add_recent(new_path)

        h, w = state.original_img.shape[:2]
        return jsonify({
            "preview": "/api/preview/result" if state.result_preview else "/api/preview/orig",
            "params": _params_to_dict(state.params, state.crop_rect),
            "crop_rect": None,  # overlay not shown — preview already reflects crop
            "filename": Path(new_path).name,
            "dims": [h, w],
            "preview_dims": list(state.preview_dims),
            "sidecar_loaded": sidecar_loaded,
            "current_index": state.current_index,
            "total_files": len(state.directory_files),
            "orientation": state.orientation,
            "flip_h": state.flip_h,
            "flip_v": state.flip_v,
        })

    @app.route("/api/carry-categories")
    def api_carry_categories():
        # Derive category→fields from per-field PARAM_CATEGORIES
        cats = {}
        for fname, cat in PARAM_CATEGORIES.items():
            cats.setdefault(cat, []).append(fname)
        return jsonify({
            "categories": cats,
            "enabled": state.carry_categories,
        })

    @app.route("/api/carry-categories", methods=["POST"])
    def api_set_carry_categories():
        data = request.get_json(force=True)
        state.carry_categories = data
        state.settings["carry_categories"] = data
        _save_settings(state.settings)
        return jsonify({"carry_categories": state.carry_categories})

    @app.route("/api/copy-settings", methods=["POST"])
    def api_copy_settings():
        """Copy settings from current image to a target image via right-click."""
        data = request.get_json(force=True)
        target = data.get("target_index")
        categories = data.get("categories", {})
        if target is None or target < 0 or target >= len(state.directory_files):
            return jsonify({"error": "Invalid target index"}), 400

        snapshot = _snapshot_carry(state)
        _auto_save(state)

        try:
            sidecar_loaded = _load_file(state, state.directory_files[target])
        except Exception as e:
            return jsonify({"error": f"Failed to read: {e}"}), 400
        state.current_index = target
        _ensure_thumb_window(target, state)

        if not sidecar_loaded:
            _apply_carry(state, snapshot, categories)
            if categories.get("geometry", False):
                _redetect_in_crop(state)

        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        _add_recent(state.directory_files[target])
        h, w = state.original_img.shape[:2]
        return jsonify({
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
            "crop_rect": None,
            "filename": Path(state.directory_files[target]).name,
            "dims": [h, w],
            "preview_dims": list(state.preview_dims),
            "sidecar_loaded": sidecar_loaded,
            "current_index": state.current_index,
            "total_files": len(state.directory_files),
            "orientation": state.orientation,
            "flip_h": state.flip_h,
            "flip_v": state.flip_v,
        })

    @app.route("/api/settings", methods=["GET"])
    def api_get_settings():
        return jsonify(state.settings)

    @app.route("/api/settings", methods=["POST"])
    def api_set_settings():
        data = request.get_json(force=True)
        for key, val in data.items():
            if key in DEFAULT_SETTINGS:
                state.settings[key] = val
        state.carry_categories = state.settings.get("carry_categories", DEFAULT_SETTINGS["carry_categories"])
        _save_settings(state.settings)
        return jsonify(state.settings)

    # ---- LUT ----
    @app.route("/api/lut", methods=["POST"])
    def api_load_lut():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400
        data = request.get_json(force=True)
        path = data.get("path", "").strip()
        if not path or not os.path.isfile(path):
            return jsonify({"error": f"LUT file not found: {path}"}), 400
        try:
            state.lut_data = parse_cube(path)
            state.lut_path = path
        except Exception as e:
            return jsonify({"error": f"Failed to parse LUT: {e}"}), 400
        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500
        _auto_save(state)
        return jsonify({
            "lut": Path(path).name,
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
        })

    @app.route("/api/lut", methods=["DELETE"])
    def api_clear_lut():
        state.lut_data = None
        state.lut_path = ""
        if state.original_img is not None:
            try:
                _run_pipeline(state)
            except Exception as e:
                return jsonify({"error": f"Inversion failed: {e}"}), 500
            _auto_save(state)
        return jsonify({
            "lut": None,
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
        })

    @app.route("/api/lut")
    def api_get_lut():
        return jsonify({"lut": Path(state.lut_path).name if state.lut_path else None})

    @app.route("/api/highlight-recovery", methods=["POST"])
    def api_toggle_highlight_recovery():
        data = request.get_json(force=True) if request.is_json else {}
        state.highlight_recovery = bool(data.get("enabled", not state.highlight_recovery))
        state.settings["highlight_recovery"] = state.highlight_recovery
        _save_settings(state.settings)
        if state.original_img is not None:
            try:
                _run_pipeline(state)
            except Exception as e:
                return jsonify({"error": f"Inversion failed: {e}"}), 500
        return jsonify({
            "highlight_recovery": state.highlight_recovery,
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
        })

    @app.route("/api/auto-wb", methods=["POST"])
    def api_auto_wb():
        """Recalculate auto WB from current image/params."""
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400
        from ..params import auto_wb as _auto_wb
        state.params.wb_high = _auto_wb(
            state.original_img, state.params.dmin, state.params.d_max,
        )
        try:
            _run_pipeline(state)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500
        state.history.push(_snapshot_params(state))
        _auto_save(state)
        return jsonify({
            "wb_high": state.params.wb_high.tolist(),
            "wb_mode": "auto",
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params, state.crop_rect),
        })

    @app.route("/api/thumb/<int:index>")
    def api_thumb(index):
        if index < 0 or index >= len(state.directory_files):
            return "", 404
        _ensure_thumb_window(index, state)
        tp = _thumb_path(state.directory_files[index])
        if tp.exists():
            return send_file(str(tp), mimetype="image/jpeg")
        return "", 404

    @app.route("/api/load-directory", methods=["POST"])
    def api_load_directory():
        """Load the first supported file from a directory."""
        data = request.get_json(force=True)
        dir_path = data.get("path", "").strip()
        if not dir_path or not os.path.isdir(dir_path):
            return jsonify({"error": f"Directory not found: {dir_path}"}), 400

        files = _scan_directory_simple(dir_path, include_subdirs=state.settings.get("include_subdirs", False))
        if not files:
            return jsonify({"error": "No supported image files found"}), 400

        try:
            sidecar_loaded = _load_file(state, files[0])
        except Exception as e:
            return jsonify({"error": f"Failed to read: {e}"}), 400

        state.directory_files = files
        state.current_index = 0
        _thumb_queued.clear()
        _ensure_thumb_window(0, state)
        _add_recent(files[0])

        h, w = state.original_img.shape[:2]
        return jsonify({
            "preview": "/api/preview/orig",
            "params": _params_to_dict(state.params, state.crop_rect),
            "crop_rect": state.crop_rect,
            "filename": Path(files[0]).name,
            "dims": [h, w],
            "preview_dims": list(state.preview_dims),
            "sidecar_loaded": sidecar_loaded,
            "orientation": state.orientation,
            "flip_h": state.flip_h,
            "flip_v": state.flip_v,
            "current_index": 0,
            "total_files": len(files),
        })

    @app.route("/api/export", methods=["POST"])
    def api_export():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400

        data = request.get_json(force=True) if request.is_json else {}
        fmt = data.get("format", "tiff16")
        quality = int(data.get("quality", 92))
        stem = Path(state.file_path).stem if state.file_path else "output"

        try:
            if state.lut_data is not None:
                saved_gamma = state.params.gamma
                saved_soft_clip = state.params.soft_clip
                state.params.gamma = 1.0
                state.params.soft_clip = 1.0
                result = invert(_get_pipeline_input(state), state.params)
                state.params.gamma = saved_gamma
                state.params.soft_clip = saved_soft_clip
                result = np.clip(result, 0.0, 1.0)
                result = apply_lut(result, state.lut_data)
            else:
                result = invert(_get_pipeline_input(state), state.params)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        # Convert Rec.2020 → sRGB for all formats (TIFF-32f keeps Rec.2020)
        if fmt == "tiff32f":
            pass  # Keep in Rec.2020 working space
        else:
            result = rec2020_to_srgb(result)
            result = np.clip(result, 0, None)

        # Apply orientation after crop, before write
        result = apply_orientation(result, state.orientation, state.flip_h, state.flip_v)

        if fmt == "tiff32f":
            suffix, dtype, mime = ".tif", "float32", "image/tiff"
            filename = f"{stem}_negconv.tif"
        elif fmt == "tiff16":
            suffix, dtype, mime = ".tif", "uint16", "image/tiff"
            filename = f"{stem}_negconv.tif"
        elif fmt == "jpeg":
            suffix, mime = ".jpg", "image/jpeg"
            filename = f"{stem}_negconv.jpg"
        elif fmt == "heic":
            suffix, mime = ".heic", "image/heic"
            filename = f"{stem}_negconv.heic"
        else:
            return jsonify({"error": f"Unknown format: {fmt}"}), 400

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            if fmt in ("tiff32f", "tiff16"):
                write_image(tmp.name, result, dtype=dtype)
            elif fmt == "jpeg":
                write_jpeg(tmp.name, result, quality=quality)
            elif fmt == "heic":
                write_heic(tmp.name, result, quality=quality)

            # Embed EXIF if available (for JPEG/HEIC only — TIFF gets no ICC)
            if state.source_exif and fmt in ("jpeg", "heic"):
                from PIL import Image as PILImage
                pil = PILImage.open(tmp.name)
                from PIL.ExifTags import Base as ExifBase
                exif = pil.getexif()
                src_exif = PILImage.Exif()
                src_exif.load(state.source_exif)
                for tag, val in src_exif.items():
                    if tag not in (0x0201, 0x0202, 0x0112):  # skip thumbnail, orientation
                        exif[tag] = val
                save_kw = {"exif": exif.tobytes()}
                if pil.info.get("icc_profile"):
                    save_kw["icc_profile"] = pil.info["icc_profile"]
                pil.save(tmp.name, pil.format, **save_kw)

            return send_file(
                tmp.name,
                as_attachment=True,
                download_name=filename,
                mimetype=mime,
            )
        except RuntimeError as e:
            if "pillow-heif" in str(e):
                return jsonify({"error": "HEIC export requires pillow-heif: pip install pillow-heif"}), 400
            raise

    return app


def _params_to_dict(params: NegconvParams, crop_rect: dict | None = None) -> dict:
    d = {
        "dmin": params.dmin.tolist(),
        "d_max": params.d_max,
        "wb_high": params.wb_high.tolist(),
        "wb_low": params.wb_low.tolist(),
        "offset": params.offset,
        "exposure": params.exposure,
        "black": params.black,
        "gamma": params.gamma,
        "soft_clip": params.soft_clip,
    }
    if crop_rect is not None:
        d["crop_rect"] = crop_rect
    return d


def _update_params_from_dict(params: NegconvParams, data: dict) -> None:
    if "dmin" in data:
        params.dmin = np.array(data["dmin"], dtype=np.float32)
    if "d_max" in data:
        params.d_max = float(data["d_max"])
    if "wb_high" in data:
        params.wb_high = np.array(data["wb_high"], dtype=np.float32)
    if "wb_low" in data:
        params.wb_low = np.array(data["wb_low"], dtype=np.float32)
    if "offset" in data:
        params.offset = float(data["offset"])
    if "exposure" in data:
        params.exposure = float(data["exposure"])
    if "black" in data:
        params.black = float(data["black"])
    if "gamma" in data:
        params.gamma = float(data["gamma"])
    if "soft_clip" in data:
        params.soft_clip = float(data["soft_clip"])


def run_gui(port: int = 5000) -> None:
    app = create_app()
    print(f"negconv GUI: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
