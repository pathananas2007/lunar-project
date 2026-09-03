"""Synthetic lunar terrain generator for MVP (no ISSDC access needed).
Kinds: crater (high texture + shadow inversion 180°), mare (low texture), sar_pair (NRD optical->SAR).
"""
import numpy as np
import cv2


def make_crater_field(h, w, n_craters=35, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float32)
    # Base regolith noise
    img = rng.normal(0.55, 0.06, (h, w)).astype(np.float32)
    # Add craters as radial gradients
    for _ in range(n_craters):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        r = int(rng.integers(12, max(13, min(h, w)//10)))
        # Crater rim bright, interior shadow dark
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        mask_inside = d < r
        mask_rim = (d >= r) & (d < r*1.25)
        # Shadow offset simulates sun angle: left side dark
        shadow = np.clip((cx - xx) / (r + 1) * 0.3, -0.3, 0.3)
        img[mask_inside] = img[mask_inside] - 0.18 - shadow[mask_inside]*0.5
        img[mask_rim] = img[mask_rim] + 0.22
    # Add rilles (linear features)
    for _ in range(3):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.line(img, (x0, y0), (x1, y1), 0.35, 1)
    # Gaussian blur for sensor PSF
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return np.clip(img, 0, 1)


def invert_shadows(img: np.ndarray):
    """Simulate 180° sun angle inversion: invert local gradients."""
    # Strong non-linear: invert around mean
    mean = float(img.mean())
    inverted = mean - (img - mean) * 0.95
    # Add slight brightness shift
    inverted = np.clip(inverted, 0, 1)
    return inverted.astype(np.float32)


def sar_simulation(optical: np.ndarray):
    """NRD: crater that is dark in optical becomes bright in SAR (roughness)."""
    # SAR speckle + log transform + invert crater contrast
    rng = np.random.default_rng(1)
    sar = 0.5 + (optical - 0.5) * -0.7  # invert contrast
    # Add speckle multiplicative noise
    speckle = rng.rayleigh(0.08, optical.shape).astype(np.float32)
    sar = sar * (1 + speckle * 0.3)
    sar = np.clip(sar, 0, 1)
    sar = cv2.GaussianBlur(sar, (0, 0), 0.8)
    return sar.astype(np.float32)


def generate_pair(kind: str = "crater", width: int = 1024, height: int = 1024):
    """
    Returns: (img1 float32 [0,1], img2 float32 [0,1]) related by homography + appearance change.
    """
    if kind == "crater":
        base = make_crater_field(height, width, seed=0)
        img1 = base
        # Apply homography (scale + translation + small rotation) to simulate different orbit
        M = cv2.getRotationMatrix2D((width/2, height/2), 2.5, 0.98)
        M[0, 2] += 12
        M[1, 2] -= 8
        img2_warped = cv2.warpAffine(base, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        img2 = invert_shadows(img2_warped)
        # Add noise
        rng = np.random.default_rng(42)
        img1 = np.clip(img1 + rng.normal(0, 0.01, img1.shape), 0, 1).astype(np.float32)
        img2 = np.clip(img2 + rng.normal(0, 0.012, img2.shape), 0, 1).astype(np.float32)
        return img1, img2

    if kind == "mare":
        rng = np.random.default_rng(7)
        base = rng.normal(0.5, 0.045, (height, width)).astype(np.float32)
        base = cv2.GaussianBlur(base, (0, 0), 2)
        # Add faint craters + subtle ridges for low-texture but matchable
        for _ in range(10):
            cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
            r = int(rng.integers(8, 18))
            cv2.circle(base, (cx, cy), r, 0.62, 1)
            cv2.circle(base, (cx, cy), r-2, 0.48, 1)
        img1 = np.clip(base, 0, 1)
        M = cv2.getRotationMatrix2D((width/2, height/2), 1.0, 1.0)
        M[0, 2] += 6; M[1, 2] += 4
        img2 = cv2.warpAffine(base, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        img2 = np.clip(img2 + rng.normal(0, 0.008, img2.shape), 0, 1).astype(np.float32)
        return img1.astype(np.float32), img2.astype(np.float32)

    if kind == "sar_pair":
        base_opt = make_crater_field(height, width, n_craters=28, seed=3)
        img1 = base_opt  # optical
        img2 = sar_simulation(base_opt)
        # Warp SAR slightly
        M = cv2.getRotationMatrix2D((width/2, height/2), -1.5, 0.97)
        M[0, 2] -= 10; M[1, 2] += 14
        img2 = cv2.warpAffine(img2, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        return img1.astype(np.float32), img2.astype(np.float32)

    # fallback random
    rng = np.random.default_rng(99)
    a = rng.random((height, width)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.02, a.shape), 0, 1).astype(np.float32)
    return a, b


def write_pds4_mock(output_dir="data/samples", width=512, height=512):
    """Write synthetic .npy + mock .xml for ingestor tests."""
    import os
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / output_dir
    base.mkdir(parents=True, exist_ok=True)
    for kind in ["crater", "mare", "sar_pair"]:
        img1, img2 = generate_pair(kind, width, height)
        np.save(str(base / f"{kind}_img1.npy"), (img1*255).astype(np.uint8))
        np.save(str(base / f"{kind}_img2.npy"), (img2*255).astype(np.uint8))
        # Mock XML
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area><product_id>ch2_{kind}_mock</product_id></Identification_Area>
  <Observation_Area><Instrument><instrument_name>TMC-2</instrument_name></Instrument></Observation_Area>
  <File_Area_Observational><Array_2D_Image><elements>{width}</elements><lines>{height}</lines></Array_2D_Image></File_Area_Observational>
</Product_Observational>"""
        (base / f"{kind}_img1.xml").write_text(xml)
        (base / f"{kind}_img2.xml").write_text(xml)
    print(f"Wrote mocks to {base}")


if __name__ == "__main__":
    write_pds4_mock()
