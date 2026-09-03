import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from generate_synthetic import generate_pair
from app.matching.pipeline import run_pipeline

def test_pipeline_crater():
    img1, img2 = generate_pair("crater", 512, 512)
    res = run_pipeline(img1, img2, algorithm="rift")
    assert res["total_keypoints"] > 0
    assert "rmse" in res
    assert res["uniformity"] >= 0

def test_pipeline_mare_loftr():
    img1, img2 = generate_pair("mare", 512, 512)
    res = run_pipeline(img1, img2, algorithm="auto")
    assert res["total_keypoints"] >= 0

def test_pipeline_sar():
    img1, img2 = generate_pair("sar_pair", 512, 512)
    res = run_pipeline(img1, img2, modality="sar", algorithm="rift")
    assert res["algorithm"] == "rift"
