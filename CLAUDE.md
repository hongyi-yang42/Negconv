# Negconv — Cineon film negative inversion pipeline

## Python environment
Always use `.venv/bin/python` directly (e.g. `.venv/bin/python -m pytest`).
Never use bare `python` or `python3` (falls through to system Python lacking deps).
Never install packages with `--break-system-packages`.

## Test fixtures
`tests/fixtures/` contains large RAW/TIF binary files for manual testing.
These are in `.gitignore` and must NEVER be committed to git.
Do not push large binary files — use `.gitignore` patterns.

## RAW config (CORRECT, don't change)
`read_raw()` uses: `user_wb=[1,1,1,1]`, `output_color=rawpy.ColorSpace.raw`,
`no_auto_bright=True`, `gamma=(1,1)`, `output_bps=16`.
sRGB matrix WITHOUT WB was tested and FAILED (crushes R channel to ~0).

## Known limitation
`detect_dmin()` auto-detect only works for flatbed scans where film base is at
the edges. Camera scans have a film holder (pure black) at the edges instead.
The function now rejects borders with max channel < 0.05 and falls back to preset.
Manual Dmin sampling via GUI eyedropper (Sprint 4) is the solution for camera scans.

## GUI architecture (Sprint 4)
Web-based: Flask + vanilla JS + HTML canvas. User develops over SSH
(Windows → Mac Mini), so native GUI frameworks won't work. Flask serves
at http://0.0.0.0:5000, user opens browser on Windows machine.
Do NOT use dearpygui, PyQt, or any framework requiring a local display.
