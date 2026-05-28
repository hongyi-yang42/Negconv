"""Sprint 11 Phase B tests: XMP sidecar, export resize, output sharpening."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest


class TestXMP:
    def test_xmp_sidecar_written(self, tmp_path):
        """Export TIFF with XMP, verify sidecar exists and contains pipeline params."""
        from negconv.io import write_xmp_sidecar
        import tifffile

        out = tmp_path / "test.tif"
        tifffile.imwrite(str(out), np.zeros((10, 10, 3), dtype=np.uint16))

        params = {
            "dmin": [1.13, 0.49, 0.27], "d_max": 1.6,
            "gamma": 4.0, "exposure": 0.92, "angle_deg": 2.5,
            "orientation": 1,
        }
        xmp_path = write_xmp_sidecar(str(out), params,
                                       source_filename="test.ARW")
        assert Path(xmp_path).exists()
        assert xmp_path.endswith(".tif.xmp")

    def test_xmp_contains_pipeline_params(self, tmp_path):
        """XMP sidecar has negconv:Pipeline namespace with dmin/gamma/exposure."""
        from negconv.io import write_xmp_sidecar
        import tifffile

        out = tmp_path / "test.tif"
        tifffile.imwrite(str(out), np.zeros((10, 10, 3), dtype=np.uint16))

        params = {"dmin": [1.13, 0.49, 0.27], "gamma": 4.0, "d_max": 1.6}
        xmp_path = write_xmp_sidecar(str(out), params, source_filename="test.ARW")

        tree = ET.parse(xmp_path)
        ns = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
        desc = tree.getroot().find(".//rdf:Description", ns)
        assert desc is not None

        # CreatorTool
        tool = desc.get("{http://ns.adobe.com/xap/1.0/}CreatorTool")
        assert tool and "Negconv" in tool

        # Pipeline params
        ns_neg = "http://negconv.org/pipeline/1.0/"
        gamma = desc.get(f"{{{ns_neg}}}gamma")
        assert gamma == "4.0"
        exposure = desc.get(f"{{{ns_neg}}}d_max")
        assert exposure == "1.6"
        dmin = desc.get(f"{{{ns_neg}}}dmin")
        assert "1.13" in dmin

        # Source filename (dc:source)
        source = desc.find("{http://purl.org/dc/elements/1.1/}source")
        assert source is not None and source.text == "test.ARW"

    def test_xmp_no_exif_still_writes(self, tmp_path):
        """XMP sidecar works when source_exif is None."""
        from negconv.io import write_xmp_sidecar
        import tifffile

        out = tmp_path / "test.tif"
        tifffile.imwrite(str(out), np.zeros((10, 10, 3), dtype=np.uint16))

        xmp_path = write_xmp_sidecar(str(out), {"gamma": 3.0})
        assert Path(xmp_path).exists()


class TestExportResize:
    def test_export_resize_long_edge(self):
        """Resize a 2000x1000 image to long edge 1024 → output is 1024x512."""
        from negconv.io import resize_for_export
        img = np.random.rand(2000, 1000, 3).astype(np.float32)
        result = resize_for_export(img, 1024)
        assert result.shape == (1024, 512, 3)

    def test_resize_no_upscale(self):
        """resize_for_export should not upscale smaller images."""
        from negconv.io import resize_for_export
        img = np.random.rand(50, 80, 3).astype(np.float32)
        result = resize_for_export(img, 200)
        assert result.shape[0] == 50
        assert result.shape[1] == 80

    def test_resize_preserves_aspect_ratio(self):
        """Aspect ratio should be maintained after resize."""
        from negconv.io import resize_for_export
        img = np.random.rand(3000, 2000, 3).astype(np.float32)
        result = resize_for_export(img, 1000)
        # 3:2 ratio, long edge 1000 → 1000x667 (rounded)
        assert abs(result.shape[0] / result.shape[1] - 1.5) < 0.01


class TestOutputSharpen:
    def test_export_sharpen_differs_from_none(self):
        """Screen sharpening should produce different pixel values than none."""
        from negconv.postproc import apply_sharpen
        img = np.random.rand(100, 100, 3).astype(np.float32) * 0.5
        none_result = img  # no sharpening
        screen = apply_sharpen(img.copy(), amount=40, radius=0.8, threshold=0.0)
        assert not np.allclose(none_result, screen, atol=0.001)

    def test_screen_vs_print_differ(self):
        """Screen and Print presets should produce different results."""
        from negconv.postproc import apply_sharpen
        img = np.random.rand(100, 100, 3).astype(np.float32) * 0.5
        screen = apply_sharpen(img.copy(), amount=40, radius=0.8, threshold=0.0)
        print_result = apply_sharpen(img.copy(), amount=80, radius=1.2, threshold=2.0)
        assert not np.allclose(screen, print_result)

    def test_sharpen_zero_amount_is_noop(self):
        """amount=0 should return the original image."""
        from negconv.postproc import apply_sharpen
        img = np.random.rand(50, 80, 3).astype(np.float32)
        result = apply_sharpen(img, amount=0)
        np.testing.assert_array_equal(result, img)
