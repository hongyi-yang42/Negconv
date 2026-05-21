from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .io import is_raw, read_image, write_image
from .params import NegconvParams, auto_detect, load_params, save_params
from .pipeline import invert

SUPPORTED_EXTENSIONS = {
    ".tif", ".tiff", ".cr2", ".cr3", ".arw", ".raf", ".nef",
    ".dng", ".orf", ".rw2", ".pef", ".srw", ".sr2",
}


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

    # White balance
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

    # JSON sidecar
    p.add_argument("--save-params", type=str, default=None, metavar="JSON")
    p.add_argument("--load-params", type=str, default=None, metavar="JSON")

    # Output format
    p.add_argument(
        "--dtype", choices=["float32", "uint16"], default="uint16",
        help="Output TIFF data type (default: uint16)",
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


def _resolve_params(args: argparse.Namespace, img: np.ndarray) -> NegconvParams:
    """Determine params from load-params, preset, or auto-detect, then apply CLI overrides."""
    if args.load_params:
        params = load_params(args.load_params)
    elif args.preset == "bw":
        params = NegconvParams.bw_film()
    elif args.preset == "color":
        params = NegconvParams.color_film()
    else:
        params = auto_detect(img)

    _apply_cli_overrides(params, args)
    return params


def _process_single(
    input_path: str, output_path: str,
    params: NegconvParams, dtype: str,
) -> None:
    """Read, invert, write a single file."""
    img = read_image(input_path)
    positive = invert(img, params)
    write_image(output_path, positive, dtype=dtype)

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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.gui:
        from .gui import run_gui
        run_gui(port=args.port)
        return

    if not args.input:
        parser.error("input is required when not using --gui")
    if not args.output:
        parser.error("-o/--output is required when not using --gui")

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Determine single vs batch mode
    if input_path.is_dir():
        # Batch mode
        inputs = _collect_inputs(args.input)
        if not inputs:
            print(f"error: no supported files found in {args.input}", file=sys.stderr)
            sys.exit(1)

        output_path.mkdir(parents=True, exist_ok=True)

        # Read first image to resolve params (auto-detect needs pixel data)
        first_img = read_image(inputs[0])
        params = _resolve_params(args, first_img)

        # Auto-save params for reproducibility
        params_file = output_path / "params.json"
        save_params(params, params_file)
        print(f"negconv {__version__}: processing {len(inputs)} files")
        print(f"  Dmin: R={params.dmin[0]:.4f} G={params.dmin[1]:.4f} B={params.dmin[2]:.4f}")
        print(f"  Dmax: {params.d_max:.3f}, gamma={params.gamma:.1f}")
        print(f"  Params saved: {params_file}")

        if args.save_params:
            save_params(params, args.save_params)

        for i, inp in enumerate(inputs, 1):
            stem = Path(inp).stem
            out_file = output_path / f"{stem}_negconv.tif"
            print(f"  Processing {i}/{len(inputs)}: {Path(inp).name}", end="")
            try:
                _process_single(inp, str(out_file), params, args.dtype)
            except Exception as e:
                print(f" ERROR: {e}", file=sys.stderr)

        print("Done.")

    else:
        # Single file mode
        img = read_image(str(input_path))
        params = _resolve_params(args, img)

        if args.save_params:
            save_params(params, args.save_params)

        positive = invert(img, params)
        write_image(str(output_path), positive, dtype=args.dtype)

        src_type = "RAW" if is_raw(str(input_path)) else "TIFF"
        print(f"negconv {__version__}: {args.input} ({src_type}) -> {args.output}")
        print(f"  Dmin: R={params.dmin[0]:.4f} G={params.dmin[1]:.4f} B={params.dmin[2]:.4f}")
        print(f"  Dmax: {params.d_max:.3f}")


if __name__ == "__main__":
    main()
