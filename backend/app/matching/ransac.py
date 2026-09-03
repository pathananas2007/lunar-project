"""RANSAC + RMSE"""
import cv2
import numpy as np


def estimate_homography(src_pts: np.ndarray, dst_pts: np.ndarray, ransac_thresh: float = 3.0):
    """
    src_pts, dst_pts: Nx2 float32
    Returns: H (3x3), mask (Nx1 bool), inlier_ratio, rmse
    Aerospace-grade: try full homography (RANSAC) + affine partial refinement, pick lowest RMSE.
    """
    if len(src_pts) < 4:
        return None, None, 0.0, float("inf")

    best = (None, None, 0.0, float("inf"))

    # 1) Full homography RANSAC
    try:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if H is not None and mask is not None:
            rmse = _rmse_for_H(H, src_pts, dst_pts, mask)
            ratio = float(mask.ravel().astype(bool).sum() / len(mask))
            if rmse < best[3]:
                best = (H, mask.ravel().astype(bool), ratio, rmse)
    except Exception:
        pass

    # 2) Affine partial 2D (more stable for orbital nadir: rotation+scale+translation)
    try:
        # Use RANSAC for affine; need at least 3 pts
        M, mask_a = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if M is not None and mask_a is not None:
            # Convert affine 2x3 to homography 3x3
            H_aff = np.eye(3, dtype=np.float64)
            H_aff[:2, :] = M
            # LMEDS refinement on inliers
            mask_b = mask_a.ravel().astype(bool)
            if mask_b.sum() >= 4:
                # Refine via least squares on inliers
                src_in = src_pts[mask_b]
                dst_in = dst_pts[mask_b]
                # Try to refine homography on inliers only with LMEDS threshold tighter
                H_ref, _ = cv2.findHomography(src_in, dst_in, 0)  # 0 = regular LS
                if H_ref is not None:
                    # Recompute mask with tighter threshold 2.0
                    mask_ref = _compute_mask(H_ref, src_pts, dst_pts, 2.0)
                    rmse_ref = _rmse_for_H(H_ref, src_pts, dst_pts, mask_ref)
                    ratio_ref = float(mask_ref.sum() / len(mask_ref))
                    if rmse_ref < best[3]:
                        best = (H_ref, mask_ref, ratio_ref, rmse_ref)
            # Also evaluate affine itself
            rmse_a = _rmse_for_H(H_aff, src_pts, dst_pts, mask_b)
            ratio_a = float(mask_b.sum() / len(mask_b))
            if rmse_a < best[3]:
                best = (H_aff, mask_b, ratio_a, rmse_a)
    except Exception:
        pass

    # 3) Try USAC_MAGSAC if available (adaptive threshold) — skip on small kp to save 2000 iters
    try:
        if hasattr(cv2, "USAC_MAGSAC") and len(src_pts) >= 20:
            # Adapt iters to kp count: fewer pts need fewer trials
            iters = 1000 if len(src_pts) < 50 else 2000
            H2, mask2 = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, ransac_thresh, maxIters=iters, confidence=0.99)
            if H2 is not None and mask2 is not None:
                rmse2 = _rmse_for_H(H2, src_pts, dst_pts, mask2)
                ratio2 = float(mask2.ravel().astype(bool).sum() / len(mask2))
                if rmse2 < best[3]:
                    best = (H2, mask2.ravel().astype(bool), ratio2, rmse2)
    except Exception:
        pass

    if best[0] is None:
        return None, None, 0.0, float("inf")
    return best


def _rmse_for_H(H, src_pts, dst_pts, mask):
    m = mask.ravel().astype(bool) if mask.ndim > 1 else mask.astype(bool)
    src_in = src_pts[m]
    dst_in = dst_pts[m]
    if len(src_in) == 0:
        return float("inf")
    ones = np.ones((len(src_in), 1), dtype=np.float64)
    src_h = np.hstack([src_in.astype(np.float64), ones])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / (proj[:, 2:3] + 1e-9)
    err = np.sqrt(((proj - dst_in.astype(np.float64)) ** 2).sum(axis=1))
    return float(np.sqrt((err ** 2).mean()))


def _compute_mask(H, src_pts, dst_pts, thresh):
    ones = np.ones((len(src_pts), 1), dtype=np.float64)
    src_h = np.hstack([src_pts.astype(np.float64), ones])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / (proj[:, 2:3] + 1e-9)
    err = np.sqrt(((proj - dst_pts.astype(np.float64)) ** 2).sum(axis=1))
    return err < thresh
