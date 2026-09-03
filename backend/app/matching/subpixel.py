"""Sub-pixel refinement via iterative gradient (cornerSubPix)."""
import cv2
import numpy as np


def refine_subpixel(image: np.ndarray, pts: list, window: int = 5, epsilon: float = 0.001, max_iter: int = 30):
    """
    image: float32 [0,1] or uint8 grayscale
    pts: list of (x,y)
    Returns: refined pts np.array Nx2 float32
    Solves sum( grad(I_p) . (q - p) ) = 0 iteratively.
    """
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    if image.dtype != np.uint8:
        img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        img_u8 = image

    corners = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    # Larger window (7x7) improves stability on crater rims; use iterative search
    win = (window, window) if window >= 7 else (7, 7)
    refined = cv2.cornerSubPix(img_u8, corners, winSize=win, zeroZone=(-1, -1), criteria=criteria)
    return refined.reshape(-1, 2)
