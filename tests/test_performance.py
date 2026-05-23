"""Performance tests: pipeline profiling, highlight recovery speed, float32 histogram."""

import time

import numpy as np
import pytest

from negconv.color import srgb_to_rec2020, rec2020_to_srgb, recover_highlights
from negconv.params import NegconvParams
from negconv.pipeline import invert


class TestPerformance:
    def test_highlight_recovery_24mp_under_5s(self):
        """Highlight recovery on a 4000x6000 image completes in <5s."""
        rng = np.random.RandomState(123)
        img = rng.rand(4000, 6000, 3).astype(np.float32) * 0.8
        # Scatter ~1000 clipped pixels
        ys = rng.randint(10, 3990, 1000)
        xs = rng.randint(10, 5990, 1000)
        channels = rng.randint(0, 3, 1000)
        for y, x, c in zip(ys, xs, channels):
            img[y, x, c] = 1.0

        t0 = time.perf_counter()
        result = recover_highlights(img, threshold=0.99)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"Highlight recovery took {elapsed:.2f}s (limit 5s)"
        assert result.shape == img.shape

    def test_pipeline_profiling_24mp(self, capsys):
        """Profile full pipeline on 6000x4000, report timing per stage."""
        rng = np.random.RandomState(42)
        raw_img = rng.rand(6000, 4000, 3).astype(np.float32) * 0.6 + 0.1
        params = NegconvParams.color_film()
        params.dmin = np.array([0.5, 0.4, 0.3], dtype=np.float32)

        times = {}

        # Stage: Rec.2020 convert (simulating TIFF input)
        t0 = time.perf_counter()
        img_rec = srgb_to_rec2020(raw_img)
        times["srgb_to_rec2020"] = time.perf_counter() - t0

        # Stage: highlight recovery
        img_rec[100, 100, 0] = 1.0  # ensure at least one clipped pixel
        t0 = time.perf_counter()
        img_recovered = recover_highlights(img_rec, threshold=0.99)
        times["highlight_recovery"] = time.perf_counter() - t0

        # Stage: pipeline inversion
        t0 = time.perf_counter()
        result = invert(img_recovered, params)
        times["pipeline_invert"] = time.perf_counter() - t0

        # Stage: Rec.2020 → sRGB
        t0 = time.perf_counter()
        result_srgb = rec2020_to_srgb(result)
        result_srgb = np.clip(result_srgb, 0, None)
        times["rec2020_to_srgb"] = time.perf_counter() - t0

        # Stage: preview JPEG generation
        t0 = time.perf_counter()
        from negconv.gui.viewer import make_preview
        jpeg = make_preview(result_srgb, 1200, quality=90)
        times["preview_jpeg"] = time.perf_counter() - t0

        total = sum(times.values())
        with capsys.disabled():
            print("\n--- Pipeline profiling (6000x4000) ---")
            for name, t in sorted(times.items(), key=lambda x: -x[1]):
                print(f"  {name:25s} {t:.3f}s  ({t/total*100:.0f}%)")
            print(f"  {'TOTAL':25s} {total:.3f}s")

        # Sanity: all stages should complete
        assert len(jpeg) > 1000
        assert result_srgb.shape == raw_img.shape
