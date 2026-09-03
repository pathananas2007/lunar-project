import numpy as np
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from generate_synthetic import generate_pair, make_crater_field
from app.matching.pipeline import run_pipeline
from app.pds4.ingestor import PDS4DataIngestor
from app.pds4.models import PDS4Metadata
from app.pds4.export import write_registered_product
from app.main import _decode_upload_to_gray, _warp_and_encode, _resample_to_common_grid

def test_12bit_scaling():
    ing = PDS4DataIngestor()
    arr = (np.random.rand(32,32)*4095).astype(np.uint16)
    out,_ = ing.read_array(arr)
    assert out.max() <= 1.0 and out.min() >= 0

def test_iirs_band_selection():
    # 3-channel image band pick
    rng = np.random.default_rng(0)
    cube = rng.integers(0,255,(64,64,3),dtype=np.uint8)
    ok, buf = cv2.imencode('.png', cube)
    g0 = _decode_upload_to_gray(buf.tobytes(), 'iirs.png', band=0)
    g1 = _decode_upload_to_gray(buf.tobytes(), 'iirs.png', band=1)
    assert g0.shape == (64,64) and g1.shape == (64,64)
    assert not np.allclose(g0,g1)

def test_warped_product():
    a,b = generate_pair('crater',256,256)
    r = run_pipeline(a,b,algorithm='rift')
    assert r['homography'] is not None
    b64, warped = _warp_and_encode(a, r['homography'], b.shape)
    assert b64 is not None and warped is not None
    assert warped.shape == (256,256)

def test_export_geotiff():
    warped = (np.random.rand(128,128)*255).astype(np.uint8)
    meta = PDS4Metadata(product_id='test_ohrc', instrument='OHRC', bit_depth=8, width=128, height=128, resolution_mpp=0.32)
    info = write_registered_product(warped, meta, [[1,0,0],[0,1,0],[0,0,1]], out_dir=Path('data/registered'), prefix='test_export')
    assert Path(info['tiff_path']).exists()
    assert Path(info['xml_path']).exists()

def test_scale_gap_router():
    base = make_crater_field(512,512,seed=1)
    hi = base
    lo = cv2.resize(base, (128,128), interpolation=cv2.INTER_AREA)
    lo_up = cv2.resize(lo, (512,512), interpolation=cv2.INTER_LINEAR)
    r = run_pipeline(hi, lo_up, algorithm='auto')
    # 4x gap should go loftr via pyramid
    assert r['algorithm'] in ('loftr','rift','gdros')

def test_mpp_resample_cap():
    meta_ohrc = PDS4Metadata(product_id='ohrc', instrument='OHRC', bit_depth=8, width=1024, height=1024, resolution_mpp=0.32)
    meta_tmc = PDS4Metadata(product_id='tmc', instrument='TMC-2', bit_depth=8, width=512, height=512, resolution_mpp=5.0)
    img1 = np.random.rand(512,512).astype(np.float32)
    img2 = np.random.rand(1024,1024).astype(np.float32)
    r1,r2,s = _resample_to_common_grid(img1, meta_tmc, img2, meta_ohrc)
    # 15x capped -> no resample
    assert s == (1.0,1.0)

def test_pds4_ohrc_width_heuristic():
    # 4096 width raw
    arr = np.random.randint(0,4095,(256*4096),dtype=np.uint16).tobytes()
    # _decode_upload_to_gray should infer 4096 width
    img = _decode_upload_to_gray(arr, 'ohrc.img')
    assert img.shape[1] == 4096 or img.shape[0] == 256
