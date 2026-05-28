from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .io import is_raw, read_image, write_image, write_jpeg, write_heic
from .params import NegconvParams, auto_detect, load_params, save_params, analyze_roll, RollProfile, detect_border_region
from .pipeline import invert
from .profiles import save_profile, load_profile, list_profiles as list_profiles_fn
from .io import RAW_EXTENSIONS

SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | {".tif", ".tiff"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="negconv",
        description="Cineon film negative inversion pipeline",
    )
    p.add_argument("input", nargs="?", help="Input file or directory")
    p.add_argument("-o", "--output", help="Output file or directory")
    p.add_argument("--version", action="version", version=f"negconv {__version__}")

    # Preset
    p.add_argument(
        "--preset", choices=["color", "bw", "auto"], default="auto",
        help="Film preset or auto-detect (default: auto)",
    )

    # Film base
    p.add_argument("--dmin-r", type=float, default=None)
    p.add_argument("--dmin-g", type=float, default=None)
    p.add_argument("--dmin-b", type=float, default=None)
    p.add_argument("--dmax", type=float, default=None)
    p.add_argument(
        "--dmin-mode", choices=["auto", "percentile", "manual"], default="auto",
        help="Dmin detection mode: auto (border→percentile→preset), percentile, manual (default: auto)",
    )

    # White balance
    p.add_argument(
        "--wb-mode", choices=["auto", "manual"], default="auto",
        help="WB mode: auto (gray-world), manual (use defaults or --wb-high-*)",
    )
    p.add_argument("--wb-high-r", type=float, default=None)
    p.add_argument("--wb-high-g", type=float, default=None)
    p.add_argument("--wb-high-b", type=float, default=None)
    p.add_argument("--wb-low-r", type=float, default=None)
    p.add_argument("--wb-low-g", type=float, default=None)
    p.add_argument("--wb-low-b", type=float, default=None)

    # Paper simulation
    p.add_argument("--exposure", type=float, default=None)
    p.add_argument("--black", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--soft-clip", type=float, default=None)
    p.add_argument("--offset", type=float, default=None)

    # Post-inversion tint
    p.add_argument("--tint", type=float, default=None,
                   help="Post-inversion tint (-1.0 green to +1.0 magenta)")

    # Post-inversion sharpening
    p.add_argument("--sharpen", type=str, default=None, metavar="AMOUNT,RADIUS,THRESHOLD",
                   help="Post-inversion unsharp mask (e.g. 50,1.0,0)")

    # Tone profile
    p.add_argument("--tone-profile", choices=["standard", "lab-warm", "lab-neutral", "cinematic"],
                   default="standard",
                   help="Built-in tone profile (default: standard)")

    # Sidecar directory (GUI-only, accepted here for consistency)
    p.add_argument("--sidecar-dir", choices=["hidden", "legacy"], default="hidden",
                   help=argparse.SUPPRESS)

    # Black level override
    p.add_argument("--black-level", type=str, default=None, metavar="R,Gr,Gb,B",
                   help="Override 4-channel black level (comma-separated)")

    # JSON sidecar
    p.add_argument("--save-params", type=str, default=None, metavar="JSON")
    p.add_argument("--load-params", type=str, default=None, metavar="JSON")

    # Named profiles
    p.add_argument("--save-profile", type=str, default=None, metavar="NAME",
                   help="Save resolved params as a named profile")
    p.add_argument("--profile", type=str, default=None, metavar="NAME",
                   help="Load a named profile instead of auto-detect")
    p.add_argument("--list-profiles", action="store_true",
                   help="List saved profiles and exit")

    # Output format
    p.add_argument(
        "--dtype", choices=["float32", "uint16"], default="uint16",
        help="Output TIFF data type (default: uint16)",
    )
    p.add_argument(
        "--format", choices=["tiff", "jpeg", "heic"], default=None,
        help="Output format (default: auto-detect from extension, or tiff)",
    )
    p.add_argument(
        "--quality", type=int, default=92,
        help="JPEG/HEIC quality 1-100 (default: 92)",
    )
    p.add_argument(
        "--resize", type=int, default=None, metavar="PIXELS",
        help="Resize longest edge to PIXELS (downscale only)",
    )
    p.add_argument(
        "--output-sharpen", choices=["none", "screen", "print"], default="none",
        help="Output sharpening preset (default: none)",
    )
    p.add_argument(
        "--no-xmp", action="store_true",
        help="Suppress XMP sidecar write on export",
    )

    # Roll analysis
    p.add_argument(
        "--roll-analyze", action="store_true",
        help="Scan directory and compute roll-wide Dmin/WB baseline before processing",
    )
    p.add_argument(
        "--roll-profile", type=str, default=None, metavar="JSON",
        help="Save/load roll profile JSON (with --roll-analyze: save; otherwise: load)",
    )

    # Border buffer
    p.add_argument(
        "--border-buffer", type=str, default="auto",
        help="Border buffer for Dmin/WB sampling: auto or N pixels (default: auto)",
    )

    # GUI mode
    p.add_argument(
        "--gui", action="store_true",
        help="Launch web GUI (Flask)",
    )
    p.add_argument(
        "--port", type=int, default=5000,
        help="Port for GUI mode (default: 5000)",
    )

    return p


def _apply_cli_overrides(params: NegconvParams, args: argparse.Namespace) -> None:
    """Apply CLI flag overrides to params (in-place)."""
    if any(v is not None for v in (args.dmin_r, args.dmin_g, args.dmin_b)):
        params.dmin = np.array([
            args.dmin_r if args.dmin_r is not None else params.dmin[0],
            args.dmin_g if args.dmin_g is not None else params.dmin[1],
            args.dmin_b if args.dmin_b is not None else params.dmin[2],
        ], dtype=np.float32)

    if args.dmax is not None:
        params.d_max = args.dmax

    if any(v is not None for v in (args.wb_high_r, args.wb_high_g, args.wb_high_b)):
        params.wb_high = np.array([
            args.wb_high_r if args.wb_high_r is not None else params.wb_high[0],
            args.wb_high_g if args.wb_high_g is not None else params.wb_high[1],
            args.wb_high_b if args.wb_high_b is not None else params.wb_high[2],
        ], dtype=np.float32)

    if any(v is not None for v in (args.wb_low_r, args.wb_low_g, args.wb_low_b)):
        params.wb_low = np.array([
            args.wb_low_r if args.wb_low_r is not None else params.wb_low[0],
            args.wb_low_g if args.wb_low_g is not None else params.wb_low[1],
            args.wb_low_b if args.wb_low_b is not None else params.wb_low[2],
        ], dtype=np.float32)

    if args.exposure is not None:
        params.exposure = args.exposure
    if args.black is not None:
        params.black = args.black
    if args.gamma is not None:
        params.gamma = args.gamma
    if args.soft_clip is not None:
        params.soft_clip = args.soft_clip
    if args.offset is not None:
        params.offset = args.offset
    if getattr(args, "tint", None) is not None:
        params.tint = args.tint


def _resolve_params(args: argparse.Namespace, img: np.ndarray) -> NegconvParams:
    """Determine params from profile, load-params, preset, or auto-detect, then apply CLI overrides."""
    if args.profile:
        profile_data = load_profile(args.profile)
        params = profile_data["params"]
    elif args.load_params:
        params = load_params(args.load_params)
    elif args.preset == "bw":
        params = NegconvParams.bw_film()
    elif args.preset == "color":
        params = NegconvParams.color_film()
    else:
        # Apply border buffer: constrain image to content region for Dmin/WB sampling
        border_px = 0
        if hasattr(args, "border_buffer") and args.border_buffer != "auto":
            border_px = int(args.border_buffer)
        sampling_img = img
        if border_px > 0:
            rect = detect_border_region(img, border_px=border_px)
            sampling_img = img[rect["y"]:rect["y"]+rect["h"], rect["x"]:rect["x"]+rect["w"]]
        params = auto_detect(sampling_img, dmin_mode=args.dmin_mode)

    # --wb-mode manual or explicit WB overrides disable auto_wb result
    if args.wb_mode == "manual" or any(
        v is not None for v in (args.wb_high_r, args.wb_high_g, args.wb_high_b)
    ):
        params.wb_high = np.ones(3, dtype=np.float32)

    _apply_cli_overrides(params, args)
    return params


def _output_format(output_path: str, fmt: str | None) -> str:
    """Determine output format from --format flag or file extension."""
    if fmt:
        return fmt
    ext = Path(output_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "jpeg"
    if ext in (".heic", ".heif"):
        return "heic"
    return "tiff"


def _write_output(output_path: str, positive: np.ndarray,
                   fmt: str, dtype: str, quality: int) -> None:
    """Write the positive image in the specified format."""
    if fmt == "jpeg":
        write_jpeg(output_path, positive, quality=quality)
    elif fmt == "heic":
        write_heic(output_path, positive, quality=quality)
    else:
        write_image(output_path, positive, dtype=dtype)


def _process_single(
    input_path: str, output_path: str,
    params: NegconvParams, dtype: str,
    fmt: str = "tiff", quality: int = 92,
    sharpen: dict | None = None,
    tone_profile: str = "standard",
    resize: int | None = None,
    output_sharpen: str = "none",
    write_xmp: bool = True,
) -> None:
    """Read, convert to Rec.2020, invert, apply post-edits, write."""
    from .color import srgb_to_rec2020, rec2020_to_srgb
    from .postproc import apply_post_edits, apply_sharpen

    img, raw_input = read_image(input_path)
    if not raw_input:
        img = srgb_to_rec2020(img)
    positive = invert(img, params)

    # Apply post-inversion edits
    positive = apply_post_edits(positive, tint=params.tint, sharpen=sharpen,
                                tone_profile=tone_profile)

    positive = rec2020_to_srgb(positive)
    positive = np.clip(positive, 0, None)

    # Export resize
    if resize and resize > 0:
        from .io import resize_for_export
        positive = resize_for_export(positive, resize)

    # Output sharpening
    if output_sharpen in ("screen", "print"):
        presets = {"screen": (40, 0.8, 0.0), "print": (80, 1.2, 2.0)}
        amt, rad, thr = presets[output_sharpen]
        positive = apply_sharpen(positive, amount=amt, radius=rad, threshold=thr)

    _write_output(output_path, positive, fmt, dtype, quality)

    # XMP sidecar
    if write_xmp:
        from .io import write_xmp_sidecar
        from .gui.app import _params_to_dict
        params_dict = {
            "dmin": params.dmin.tolist(), "d_max": params.d_max,
            "wb_high": params.wb_high.tolist(), "wb_low": params.wb_low.tolist(),
            "exposure": params.exposure, "gamma": params.gamma,
            "black": params.black, "soft_clip": params.soft_clip,
            "offset": params.offset, "tint": params.tint,
            "tone_profile": tone_profile,
        }
        exif_bytes = extract_exif(input_path)
        write_xmp_sidecar(output_path, params_dict,
                           source_exif=exif_bytes,
                           source_filename=Path(input_path).name)

    src_type = "RAW" if is_raw(input_path) else "TIFF"
    print(f"  {Path(input_path).name} ({src_type}) -> {Path(output_path).name}")


def _collect_inputs(input_path: str) -> list[str]:
    """Gather all supported files from a directory."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = []
        for ext in sorted(SUPPORTED_EXTENSIONS):
            files.extend(glob.glob(str(p / f"*{ext}")))
            files.extend(glob.glob(str(p / f"*{ext.upper()}")))
        return sorted(set(files))
    return [str(p)]


def _parse_sharpen(sharpen_str: str | None) -> dict | None:
    """Parse --sharpen AMOUNT,RADIUS,THRESHOLD into a dict."""
    if sharpen_str is None:
        return None
    parts = sharpen_str.split(",")
    if len(parts) != 3:
        print("error: --sharpen requires AMOUNT,RADIUS,THRESHOLD", file=sys.stderr)
        sys.exit(1)
    return {
        "amount": float(parts[0]),
        "radius": float(parts[1]),
        "threshold": float(parts[2]),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.gui:
        from .gui import run_gui
        run_gui(port=args.port)
        return

    sharpen = _parse_sharpen(args.sharpen)

    if args.list_profiles:
        profiles = list_profiles_fn()
        if not profiles:
            print("No saved profiles.")
        for p in profiles:
            print(f"  {p['name']}  ({p['created'][:10]})")
        return

    if not args.input:
        parser.error("input is required when not using --gui")
    if not args.output:
        parser.error("-o/--output is required when not using --gui")

    input_path = Path(args.input)
    output_path = Path(args.output)
    fmt = _output_format(str(output_path), args.format)
    fmt_ext = {"tiff": ".tif", "jpeg": ".jpg", "heic": ".heic"}[fmt]

    # Determine single vs batch mode
    if input_path.is_dir():
        # Batch mode
        inputs = _collect_inputs(args.input)
        if not inputs:
            print(f"error: no supported files found in {args.input}", file=sys.stderr)
            sys.exit(1)

        output_path.mkdir(parents=True, exist_ok=True)

        # Roll analysis: compute roll-wide baseline before processing
        roll_profile = None
        if args.roll_analyze:
            print(f"  Analyzing roll ({len(inputs)} frames)...", end="", flush=True)
            images = []
            for inp in inputs:
                img, _ = read_image(inp)
                images.append(img)

            def _roll_progress(cur, tot):
                pct = int(cur / tot * 100) if tot else 0
                print(f"\r  Analyzing roll ({cur}/{tot}, {pct}%)...", end="", flush=True)

            roll_profile = analyze_roll(images, progress_callback=_roll_progress)
            print(f"\r  Roll analysis: {roll_profile.num_frames} frames, "
                  f"{len(roll_profile.outlier_indices)} outliers")

            if args.roll_profile:
                roll_profile.save(args.roll_profile)
                print(f"  Roll profile saved: {args.roll_profile}")
            else:
                rp_dir = Path(args.input) / ".negconv"
                rp_dir.mkdir(parents=True, exist_ok=True)
                roll_profile.save(rp_dir / "roll_profile.json")

            # Use roll baseline as Dmin/WB
            print(f"  Roll Dmin: R={roll_profile.roll_dmin[0]:.4f} "
                  f"G={roll_profile.roll_dmin[1]:.4f} B={roll_profile.roll_dmin[2]:.4f}")

        elif args.roll_profile:
            roll_profile = RollProfile.load(args.roll_profile)
            print(f"  Loaded roll profile: {roll_profile.num_frames} frames")

        # Read first image to resolve params (auto-detect needs pixel data)
        first_img, _ = read_image(inputs[0])
        params = _resolve_params(args, first_img)

        # Override with roll baseline if available
        if roll_profile:
            params.dmin = roll_profile.roll_dmin.copy()
            params.wb_high = roll_profile.roll_wb_high.copy()

        # Auto-save params for reproducibility
        params_file = output_path / "params.json"
        save_params(params, params_file)
        print(f"negconv {__version__}: processing {len(inputs)} files")
        print(f"  Dmin: R={params.dmin[0]:.4f} G={params.dmin[1]:.4f} B={params.dmin[2]:.4f}")
        print(f"  Dmax: {params.d_max:.3f}, gamma={params.gamma:.1f}")
        print(f"  Params saved: {params_file}")

        if args.save_params:
            save_params(params, args.save_params)
        if args.save_profile:
            save_profile(args.save_profile, params)
            print(f"  Profile saved: {args.save_profile}")

        for i, inp in enumerate(inputs, 1):
            stem = Path(inp).stem
            out_file = output_path / f"{stem}_negconv{fmt_ext}"
            print(f"  Processing {i}/{len(inputs)}: {Path(inp).name}", end="")
            try:
                _process_single(inp, str(out_file), params, args.dtype, fmt, args.quality,
                                sharpen=sharpen, tone_profile=args.tone_profile,
                                resize=args.resize, output_sharpen=args.output_sharpen,
                                write_xmp=not args.no_xmp)
            except Exception as e:
                print(f" ERROR: {e}", file=sys.stderr)

        print("Done.")

    else:
        # Single file mode
        img, _ = read_image(str(input_path))
        params = _resolve_params(args, img)

        if args.save_params:
            save_params(params, args.save_params)
        if args.save_profile:
            save_profile(args.save_profile, params, tone_profile=args.tone_profile)
            print(f"  Profile saved: {args.save_profile}")

        _process_single(str(input_path), str(output_path), params, args.dtype, fmt,
                        args.quality, sharpen=sharpen, tone_profile=args.tone_profile,
                        resize=args.resize, output_sharpen=args.output_sharpen,
                        write_xmp=not args.no_xmp)

        src_type = "RAW" if is_raw(str(input_path)) else "TIFF"
        print(f"negconv {__version__}: {args.input} ({src_type}) -> {args.output}")
        print(f"  Dmin: R={params.dmin[0]:.4f} G={params.dmin[1]:.4f} B={params.dmin[2]:.4f}")
        print(f"  Dmax: {params.d_max:.3f}")


if __name__ == "__main__":
    main()
