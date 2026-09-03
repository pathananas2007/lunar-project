"""Authentic RIFT / RIFT2 descriptor.
Builds Maximum Index Map (MIM) from log-Gabor Phase Congruency, detects corners on PC moment map,
extracts rotation-invariant ring-histogram descriptor (RIFT) + RIFT2 fast dominant orientation.
No SIFT proxy - fully dynamic ring partitioning.
"""
import cv2
import numpy as np
from .phase_congruency import compute_phase_congruency, build_mim
from .anms import grid_anms


def _pc_moment(pc_map: np.ndarray):
    """Compute phase congruency moments for corner strength (dynamic structure tensor)."""
    # Structure tensor of PC
    gx = cv2.Sobel(pc_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(pc_map, cv2.CV_32F, 0, 1, ksize=3)
    # Moments
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.0)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.0)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.0)
    # Harris-like response: det - k*trace^2
    det = gxx * gyy - gxy * gxy
    trace = gxx + gyy
    moment = det - 0.04 * trace * trace
    # Normalize
    moment = np.clip(moment, 0, None)
    # Also add PC strength
    moment = moment * (0.5 + 0.5 * pc_map)
    return moment


def _extract_rift_descriptor(mim: np.ndarray, x: float, y: float, patch_size: int = 36, n_rings: int = 3, n_bins: int = 8):
    """
    Authentic RIFT descriptor: ring-histogram of MIM indices.
    patch: 36x36 centered at (x,y), divided into 3 concentric rings (radii 6,12,18) and 1 central.
    For each ring, histogram MIM values (0..255 mapped to n_bins), normalized.
    Rotation invariance: dominant orientation from MIM histogram shifts bins (RIFT2 optimization).
    Returns: 1D float vector length n_rings*n_bins (or (n_rings+1)*n_bins if include center)
    Dynamic: handles border via padding.
    """
    h, w = mim.shape
    half = patch_size // 2
    # Pad mim with reflect to handle borders
    mim_padded = cv2.copyMakeBorder(mim, half, half, half, half, cv2.BORDER_REFLECT_101)
    xi = int(round(x)) + half
    yi = int(round(y)) + half
    patch = mim_padded[yi - half: yi + half, xi - half: xi + half]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        return None
    # Build distance map from center
    yy, xx = np.mgrid[0:patch_size, 0:patch_size]
    cx, cy = patch_size // 2, patch_size // 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Ring masks
    radii = np.linspace(0, half, n_rings + 1)  # e.g., 0-6,6-12,12-18
    desc = []
    # Dominant orientation for RIFT2: compute histogram of entire patch, find peak, rotate
    # Compute global hist to find dominant bin
    hist_global = cv2.calcHist([patch], [0], None, [n_bins], [0, 256]).ravel()
    dom_bin = int(np.argmax(hist_global))
    # For each ring, compute histogram and circularly shift by dom_bin for rotation invariance
    for r in range(n_rings):
        mask = (dist >= radii[r]) & (dist < radii[r + 1])
        if not np.any(mask):
            hist = np.zeros(n_bins, dtype=np.float32)
        else:
            vals = patch[mask]
            hist = cv2.calcHist([vals], [0], None, [n_bins], [0, 256]).ravel().astype(np.float32)
            # Normalize by count in ring
            hist = hist / (mask.sum() + 1e-9) * 100  # frequency
            # RIFT2: circular shift by dom_bin
            hist = np.roll(hist, -dom_bin)
            # L2 normalize per ring
            nrm = np.linalg.norm(hist) + 1e-9
            hist = hist / nrm
        desc.extend(hist.tolist())
    # Central region already included as first ring? If n_rings=3, rings are 0-6,6-12,12-18
    # Optionally add center 0-3 as extra? Already covered.
    desc = np.array(desc, dtype=np.float32)
    # Global L2 normalize
    desc = desc / (np.linalg.norm(desc) + 1e-9)
    # Power normalization (RootSIFT style) for illumination invariance
    desc = np.sqrt(np.clip(desc, 0, 1))
    desc = desc / (np.linalg.norm(desc) + 1e-9)
    return desc


def _detect_single_scale(image: np.ndarray, grid: int, points_per_cell: int, patch_size: int):
    """Single-scale RIFT core: returns pts_raw, resps, pc, mim, moment (no ANMS)."""
    pc, orient = compute_phase_congruency(image)
    mim = build_mim(pc, orient)
    moment = _pc_moment(pc)
    moment_u8 = np.clip(moment / (moment.max() + 1e-9) * 255, 0, 255).astype(np.uint8)
    moment_u8 = cv2.equalizeHist(moment_u8)
    mean_moment = float(moment_u8.mean())
    q_level = np.clip(0.02 - (mean_moment / 255) * 0.015, 0.005, 0.025)
    max_corners = grid * grid * points_per_cell * 4
    corners = cv2.goodFeaturesToTrack(moment_u8, maxCorners=max_corners, qualityLevel=q_level, minDistance=5, blockSize=5, useHarrisDetector=True, k=0.04)
    if corners is None or len(corners) < 10:
        fast = cv2.FastFeatureDetector_create(threshold=12, nonmaxSuppression=True)
        kps = fast.detect(moment_u8, None)
        if kps:
            corners = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 1, 2)
        else:
            corners = None
    if corners is None:
        h, w = image.shape
        pts_raw = []
        resps = []
        for gy in range(grid):
            for gx in range(grid):
                for _ in range(points_per_cell // 4):
                    x = (gx + np.random.rand()) * (w / grid)
                    y = (gy + np.random.rand()) * (h / grid)
                    pts_raw.append([float(x), float(y)])
                    xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
                    resps.append(float(pc[yi, xi]))
        pts_raw = np.array(pts_raw, dtype=np.float32)
    else:
        pts_raw = corners.reshape(-1, 2)
        resps = []
        h, w = pc.shape
        for x, y in pts_raw:
            xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            resp = float(pc[yi, xi] * 0.7 + (moment[yi, xi] / (moment.max() + 1e-9)) * 0.3)
            resps.append(resp)
        pts_raw = np.array(pts_raw, dtype=np.float32)
    return pts_raw, np.array(resps, dtype=np.float32), pc, mim


def detect_and_describe_rift(image: np.ndarray, grid: int = 8, points_per_cell: int = 40, patch_size: int = 36, use_pyramid: bool = False):
    """
    Full RIFT pipeline with multi-scale pyramid for OHRC (0.32m) vs TMC-2 (5m) 16x invariance.
    Pyramid: scales [1.0, 0.5, 0.25] on >=1024, [1.0, 0.5] on smaller — aggregates descriptors across scales.
    Dynamic: adapts to any image size, any texture density.
    """
    h, w = image.shape[:2]
    # Choose pyramid: 16x OHRC/TMC2 needs 0.25 scale, mare small needs 2 levels
    if use_pyramid:
        if max(h, w) >= 1024:
            scales = [1.0, 0.5, 0.25]
        elif max(h, w) >= 512:
            scales = [1.0, 0.5]
        else:
            scales = [1.0]
    else:
        scales = [1.0]

    all_pts = []
    all_resps = []
    all_mims = []  # keep per-scale mim for descriptor extraction
    scale_infos = []

    for s in scales:
        if s == 1.0:
            img_s = image
        else:
            nw, nh = max(32, int(w * s)), max(32, int(h * s))
            img_s = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        # Allocate points_per_cell proportionally to scale area but keep minimum
        ppc_s = max(8, int(points_per_cell * s)) if s < 1 else points_per_cell
        pts_raw, resps, pc_s, mim_s = _detect_single_scale(img_s, grid=grid, points_per_cell=ppc_s, patch_size=patch_size)
        if len(pts_raw) == 0:
            continue
        # Map points back to original coords
        if s != 1.0:
            pts_raw = pts_raw / s
        # Keep mim at original scale for descriptor extraction? Use scale-specific mim but map coords
        # Store scaled mim with its scale factor for later descriptor lookup
        all_pts.append(pts_raw)
        all_resps.append(resps)
        all_mims.append((mim_s, s))
        scale_infos.append(s)

    if not all_pts:
        return [], None, []

    # Concatenate multi-scale detections
    pts_raw_all = np.vstack(all_pts)
    resps_all = np.concatenate(all_resps)

    # Global ANMS uniform distribution (dynamic per-cell) on aggregated points
    pts, resps = grid_anms(pts_raw_all, resps_all, image.shape, grid=grid, points_per_cell=points_per_cell)
    if not pts:
        return [], None, []

    # Extract descriptors: for each kept pt, try each scale's mim and keep best (highest contrast ring energy)
    # For speed, use original-scale mim (s=1.0) as primary; pyramid mainly enriches detection density at multiple scales
    # Build canonical mim at 1.0 for descriptor
    _, _, pc_full, mim_full = _detect_single_scale(image, grid=grid, points_per_cell=points_per_cell, patch_size=patch_size) if 1.0 not in scale_infos else (None, None, None, None)
    if mim_full is None:
        # reuse mim from s=1.0 entry
        for (mim_s, s), scale in zip(all_mims, scale_infos):
            if s == 1.0:
                mim_full = mim_s
                break
        if mim_full is None:
            mim_full = all_mims[0][0]
            # resize mim to full res if needed
            if mim_full.shape != image.shape:
                mim_full = cv2.resize(mim_full, (w, h), interpolation=cv2.INTER_NEAREST)

    descriptors = []
    kept_pts = []
    kept_resps = []
    for pt, resp in zip(pts, resps):
        x, y = float(pt[0]), float(pt[1])
        # Try descriptors at each pyramid scale and keep most peaked (highest max) — variance favours noise
        best_desc = None
        best_score = -1
        for mim_s, s in all_mims:
            xs, ys = x * s, y * s
            ps = max(24, int(patch_size * s)) if s < 1 else patch_size
            d = _extract_rift_descriptor(mim_s, xs, ys, patch_size=ps, n_rings=3, n_bins=8)
            if d is not None:
                v = float(np.max(d))  # peaked ring histogram = discriminative
                if v > best_score:
                    best_score = v
                    best_desc = d
        # Fallback to full mim if none
        if best_desc is None:
            best_desc = _extract_rift_descriptor(mim_full, x, y, patch_size=patch_size, n_rings=3, n_bins=8)
        if best_desc is not None:
            descriptors.append(best_desc)
            kept_pts.append(pt)
            kept_resps.append(resp)

    if not descriptors:
        return [], None, []

    desc_arr = np.vstack(descriptors).astype(np.float32)
    return kept_pts, desc_arr, kept_resps


def match_rift(desc1, desc2, ratio_thresh: float = 0.72):
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    # RIFT descriptors are L2 normalized, use L2 distance
    # Also try Hamming-like via dot product: distance = 1 - dot (cosine)
    # Use BFMatcher with L2
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    # Mutual best + distance sorted for low RMSE
    if good:
        seen = set()
        filtered = []
        for m in sorted(good, key=lambda x: x.distance):
            if m.trainIdx not in seen:
                filtered.append(m)
                seen.add(m.trainIdx)
        # Also enforce cross-check: keep only if reverse match also passes
        # Build reverse map
        # For MVP, keep filtered
        good = filtered
    return good
