from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from ..io import is_raw, read_image, write_image
from ..params import NegconvParams, auto_detect, save_params, load_params
from ..pipeline import invert
from .viewer import make_preview

PREVIEW_MAX_WIDTH = 1200


@dataclass
class GuiState:
    original_img: np.ndarray | None = None
    params: NegconvParams = field(default_factory=NegconvParams.color_film)
    file_path: str = ""
    orig_preview: bytes = b""
    result_preview: bytes = b""
    preview_dims: tuple[int, int] = (0, 0)  # (width, height) of preview image


def _sample_dmin(img: np.ndarray, orig_x: int, orig_y: int, patch: int = 5) -> np.ndarray:
    """Sample a patch_size x patch_size area around (orig_x, orig_y) and average per-channel."""
    h, w = img.shape[:2]
    half = patch // 2
    y0 = max(orig_y - half, 0)
    y1 = min(orig_y + half + 1, h)
    x0 = max(orig_x - half, 0)
    x1 = min(orig_x + half + 1, w)
    region = img[y0:y1, x0:x1, :]
    return np.mean(region, axis=(0, 1)).astype(np.float32)


def _preview_to_orig_coords(
    px: int, py: int,
    preview_w: int, preview_h: int,
    orig_w: int, orig_h: int,
) -> tuple[int, int]:
    """Map preview pixel coords to original image pixel coords."""
    orig_x = int(px * orig_w / preview_w)
    orig_y = int(py * orig_h / preview_h)
    return (
        min(max(orig_x, 0), orig_w - 1),
        min(max(orig_y, 0), orig_h - 1),
    )


def create_app() -> Flask:
    app = Flask(__name__)
    state = GuiState()

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
            img = read_image(path)
        except Exception as e:
            return jsonify({"error": f"Failed to read: {e}"}), 400

        state.original_img = img
        state.file_path = path
        state.params = auto_detect(img)

        state.orig_preview = make_preview(img, PREVIEW_MAX_WIDTH)

        # Compute preview dimensions for coordinate mapping
        from PIL import Image as PILImage
        pil_preview = PILImage.open(
            __import__("io").BytesIO(state.orig_preview)
        )
        state.preview_dims = (pil_preview.width, pil_preview.height)

        h, w = img.shape[:2]
        return jsonify({
            "preview": "/api/preview/orig",
            "params": _params_to_dict(state.params),
            "filename": Path(path).name,
            "dims": [h, w],
            "preview_dims": list(state.preview_dims),
        })

    @app.route("/api/preview/orig")
    def api_preview_orig():
        if not state.orig_preview:
            return "", 404
        return send_file(
            __import__("io").BytesIO(state.orig_preview),
            mimetype="image/jpeg",
        )

    @app.route("/api/preview/result")
    def api_preview_result():
        if not state.result_preview:
            return "", 404
        return send_file(
            __import__("io").BytesIO(state.result_preview),
            mimetype="image/jpeg",
        )

    @app.route("/api/invert", methods=["POST"])
    def api_invert():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400

        try:
            result = invert(state.original_img, state.params)
            state.result_preview = make_preview(result, PREVIEW_MAX_WIDTH)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        return jsonify({
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params),
        })

    @app.route("/api/pick-dmin", methods=["POST"])
    def api_pick_dmin():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400

        data = request.get_json(force=True)
        px, py = data.get("x", 0), data.get("y", 0)

        h, w = state.original_img.shape[:2]
        preview_w, preview_h = state.preview_dims
        if preview_w == 0 or preview_h == 0:
            return jsonify({"error": "No preview loaded"}), 400

        orig_x, orig_y = _preview_to_orig_coords(
            px, py, preview_w, preview_h, w, h,
        )

        dmin = _sample_dmin(state.original_img, orig_x, orig_y)
        state.params.dmin = dmin

        # Auto-invert with new Dmin
        try:
            result = invert(state.original_img, state.params)
            state.result_preview = make_preview(result, PREVIEW_MAX_WIDTH)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        return jsonify({
            "dmin": dmin.tolist(),
            "preview": "/api/preview/result",
            "params": _params_to_dict(state.params),
        })

    @app.route("/api/params", methods=["GET"])
    def api_get_params():
        return jsonify(_params_to_dict(state.params))

    @app.route("/api/params", methods=["POST"])
    def api_set_params():
        data = request.get_json(force=True)
        _update_params_from_dict(state.params, data)
        return jsonify(_params_to_dict(state.params))

    @app.route("/api/preset/<name>", methods=["POST"])
    def api_preset(name):
        if name == "color":
            state.params = NegconvParams.color_film()
        elif name == "bw":
            state.params = NegconvParams.bw_film()
        else:
            return jsonify({"error": f"Unknown preset: {name}"}), 400
        return jsonify(_params_to_dict(state.params))

    @app.route("/api/export", methods=["POST"])
    def api_export():
        if state.original_img is None:
            return jsonify({"error": "No image loaded"}), 400

        data = request.get_json(force=True) if request.is_json else {}
        dtype = data.get("dtype", "uint16")

        try:
            result = invert(state.original_img, state.params)
        except Exception as e:
            return jsonify({"error": f"Inversion failed: {e}"}), 500

        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        try:
            write_image(tmp.name, result, dtype=dtype)
            stem = Path(state.file_path).stem if state.file_path else "output"
            filename = f"{stem}_negconv.tif"
            return send_file(
                tmp.name,
                as_attachment=True,
                download_name=filename,
                mimetype="image/tiff",
            )
        finally:
            # Clean up after response is sent
            pass

    return app


def _params_to_dict(params: NegconvParams) -> dict:
    return {
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
