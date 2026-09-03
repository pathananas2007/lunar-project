"""Hybrid Pipeline Orchestrator — routes RIFT vs LightGlue/LoFTR + subpixel + RANSAC"""
import time
import cv2
import numpy as np
from ..preprocessing.clahe import apply_clahe
from .rift import detect_and_describe_rift, match_rift
from .loftr_lightglue import detect_and_match_lightglue
from .gdros import dense_registration_gdros
from .subpixel import refine_subpixel
from .ransac import estimate_homography
from .anms import uniformity_score


def texture_score(image: np.ndarray) -> float:
    """Variance of Laplacian as texture metric."""
    if image.dtype != np.uint8:
        u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        u8 = image
    return float(cv2.Laplacian(u8, cv2.CV_64F).var())


def run_pipeline(
    img1: np.ndarray,
    img2: np.ndarray,
    modality: str = "auto",
    algorithm: str = "auto",
    ransac_thresh: float = 3.0,
    grid: int = 8,
    points_per_cell: int = 40,
    clahe_clip: float = 2.0,
) -> dict:
    """
    img1,img2: float32 [0,1] grayscale HxW
    algorithm: auto|rift|loftr|lightglue
    modality: auto|optical|sar|infrared (hints router)
    """
    t0 = time.perf_counter()

    # 1. CLAHE
    c1 = apply_clahe(img1, clip_limit=clahe_clip, tile_grid=8)
    c2 = apply_clahe(img2, clip_limit=clahe_clip, tile_grid=8)

    # 2. Router — ISRO fix: scale-aware (OHRC 0.32m vs TMC2 5m =16x) — scale FIRST, then modality
    chosen = algorithm
    if algorithm == "auto":
        h1, w1 = c1.shape[:2]; h2, w2 = c2.shape[:2]
        scale_gap = max(h1/h2, h2/h1, w1/w2, w2/w1) if min(h2,w2)>0 else 1.0
        if scale_gap >= 3:
            chosen = "loftr"  # dense handles large scale (15x OHRC/TMC2)
        elif modality.lower() in ("sar", "dfsar", "infrared", "iirs"):
            chosen = "rift"  # NRD invariant for SAR/IR
        else:
            tx1 = texture_score(c1)
            tx2 = texture_score(c2)
            if min(tx1, tx2) < 80:
                chosen = "loftr"
            else:
                chosen = "rift"

    src_pts = []
    dst_pts = []
    confidences = []

    if chosen == "rift":
        # Scale-aware pyramid: enable only when resolution gap large (OHRC vs TMC2)
        h1, w1 = c1.shape[:2]; h2, w2 = c2.shape[:2]
        gap = max(h1/h2, h2/h1, w1/w2, w2/w1) if min(h2,w2)>0 else 1.0
        use_pyr = gap >= 2.0
        pts1, desc1, _ = detect_and_describe_rift(c1, grid=grid, points_per_cell=points_per_cell, use_pyramid=use_pyr)
        pts2, desc2, _ = detect_and_describe_rift(c2, grid=grid, points_per_cell=points_per_cell, use_pyramid=use_pyr)
        matches = match_rift(desc1, desc2)
        if matches:
            src_pts = [pts1[m.queryIdx] for m in matches]
            dst_pts = [pts2[m.trainIdx] for m in matches]
            # Dynamic confidence from descriptor distance (live, not static)
            confidences = [float(np.clip(1.0 - m.distance / 2.0, 0.1, 0.99)) for m in matches]
    elif chosen == "gdros":
        # GDROS dense geometry-guided path for large SAR-optical transforms
        pts1, pts2, confs = dense_registration_gdros(c1, c2, max_iter=4)
        src_pts, dst_pts, confidences = pts1, pts2, confs
        chosen = "gdros"
    else:
        # Dynamic router: loftr / lightglue (real Kornia models, CPU auto)
        # Try LightGlue first (sparse, fast), fallback to LoFTR dense via confidence adapt
        norm_name = "lightglue" if chosen == "lightglue" else "loftr"
        pts1, pts2, confs = detect_and_match_lightglue(c1, c2, conf_thresh=0.45)
        if len(pts1) < 8:
            pts1, pts2, confs = detect_and_match_lightglue(c1, c2, conf_thresh=0.30)
        # If still sparse and modality is SAR/mixed, try GDROS dense as final dynamic fallback
        if len(pts1) < 8 and modality.lower() in ("sar","dfsar","infrared","iirs"):
            pts1, pts2, confs = dense_registration_gdros(c1, c2, max_iter=3)
            chosen = "gdros"
        else:
            chosen = norm_name
        src_pts, dst_pts, confidences = pts1, pts2, confs

    total_kp = len(src_pts)
    if total_kp == 0:
        return {
            "algorithm": chosen,
            "total_keypoints": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rmse": 999.0,
            "homography": None,
            "correspondences": [],
            "uniformity": 0.0,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "texture_scores": [float(texture_score(c1)), float(texture_score(c2))],
            "warped_b64": None,
            "warped_shape": None,
        }

    # 3. Subpixel refinement (both images)
    src_arr = np.array(src_pts, dtype=np.float32)
    dst_arr = np.array(dst_pts, dtype=np.float32)
    src_ref = refine_subpixel(c1, src_arr.tolist())
    dst_ref = refine_subpixel(c2, dst_arr.tolist())

    # 4. RANSAC
    H, mask, inlier_ratio, rmse = estimate_homography(src_ref, dst_ref, ransac_thresh=ransac_thresh)

    # Uniformity on inliers
    inlier_pts = src_ref[mask] if mask is not None and H is not None else src_ref
    uniform = uniformity_score(inlier_pts.tolist() if len(inlier_pts) else [], c1.shape, grid=grid)

    # Build correspondences array for frontend (inliers + outliers color-coded)
    correspondences = []
    if mask is not None:
        for i, (s, d) in enumerate(zip(src_ref, dst_ref)):
            correspondences.append({
                "x1": float(s[0]), "y1": float(s[1]),
                "x2": float(d[0]), "y2": float(d[1]),
                "inlier": bool(mask[i]),
                "confidence": float(confidences[i]) if i < len(confidences) else 0.5,
            })
    else:
        for s, d, cf in zip(src_ref, dst_ref, confidences):
            correspondences.append({"x1": float(s[0]), "y1": float(s[1]), "x2": float(d[0]), "y2": float(d[1]), "inlier": False, "confidence": float(cf)})

    latency = (time.perf_counter() - t0) * 1000

    return {
        "algorithm": chosen,
        "total_keypoints": total_kp,
        "inlier_count": int(mask.sum()) if mask is not None else 0,
        "inlier_ratio": float(inlier_ratio),
        "rmse": float(rmse) if np.isfinite(rmse) else 999.0,
        "homography": H.tolist() if H is not None else None,
        "correspondences": correspondences,
        "uniformity": float(uniform),
        "latency_ms": float(latency),
        "texture_scores": [float(texture_score(c1)), float(texture_score(c2))],
    }
