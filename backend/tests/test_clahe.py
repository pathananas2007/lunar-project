import numpy as np
from app.preprocessing.clahe import apply_clahe

def test_clahe_float():
    img = np.random.rand(128,128).astype(np.float32)
    out = apply_clahe(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1.0
