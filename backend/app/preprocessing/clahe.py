"""CLAHE preprocessing - FR-02 dynamic (tileGrid adapts to image size)."""
import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Dynamic CLAHE: tileGrid auto-scales to image size, clipLimit modulates per image variance."""
    h, w = image.shape[:2]
    # Dynamic tile: ensure at least 32px per tile, adapt grid to image
    # For 1024, 8x8 => 128px tiles; for 512, 8x8 => 64px; for 256, auto reduce to 4x4
    if min(h, w) < 400 and tile_grid > 4:
        tile_grid = 4
    elif min(h, w) > 1500 and tile_grid < 16:
        tile_grid = 16

    # Dynamic clip: low-contrast mare needs lower clip to avoid noise amplification
    # ISRO fix: blended clip was double-applied; use max(caller, dynamic) so caller can force higher
    if image.dtype == np.uint8:
        std = float(image.std())
        dyn_clip = np.clip(1.5 + (std / 40) * 1.0, 1.2, 3.0)
        clip_limit = float(max(clip_limit, dyn_clip)) if clip_limit else float(dyn_clip)
    else:
        std = float((image * 255).std())
        dyn_clip = np.clip(1.5 + (std / 40) * 1.0, 1.2, 3.0)
        clip_limit = float(max(clip_limit, dyn_clip)) if clip_limit else float(dyn_clip)

    if image.dtype != np.uint8:
        img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        out_u8 = clahe.apply(img_u8)
        return out_u8.astype(np.float32) / 255.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    out = clahe.apply(image)
    return out.astype(np.float32) / 255.0 if out.dtype == np.uint8 else out


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)
