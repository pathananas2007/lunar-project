import numpy as np
from app.pds4.ingestor import PDS4DataIngestor

def test_normalize():
    ing = PDS4DataIngestor()
    arr_uint8 = (np.random.rand(32,32)*255).astype(np.uint8)
    out,_ = ing.read_array(arr_uint8)
    assert out.max() <= 1.0 and out.min() >= 0
    assert out.dtype == np.float32

def test_synthetic_array():
    ing = PDS4DataIngestor()
    arr = np.random.rand(64,64).astype(np.float32)
    out, meta = ing.read_array(arr)
    assert out.shape == (64,64)
    assert meta.instrument == "TMC-2"
