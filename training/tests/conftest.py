import numpy as np
import pytest

from lasr import ffmpeg


def needs_codecs(*names: str):
    have = ffmpeg.available_codecs()
    missing = [n for n in names if n not in have]
    return pytest.mark.skipif(bool(missing), reason=f"ffmpeg encoders missing: {missing}")


@pytest.fixture
def hr_clip() -> np.ndarray:
    """A small textured clip: (T=3, 64, 64, 3) float RGB in [0, 1]."""
    from lasr.synthetic import synthetic_image

    rng = np.random.default_rng(0)
    img = synthetic_image(96, rng)
    return np.stack([img[i : i + 64, i : i + 64] for i in range(3)]).astype(np.float32)
