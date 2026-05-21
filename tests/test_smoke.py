"""Smoke test: import negconv, assert version string exists."""


def test_version():
    import negconv

    assert hasattr(negconv, "__version__")
    assert isinstance(negconv.__version__, str)
    assert len(negconv.__version__) > 0


def test_imports():
    from negconv.pipeline import invert
    from negconv.params import NegconvParams
    from negconv.io import read_image, write_image

    assert callable(invert)
    assert callable(read_image)
    assert callable(write_image)
