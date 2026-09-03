import numpy as np

def to_float01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    m = float(arr.max()) if arr.size else 1.0
    if m > 1.5:
        if m > 4095:
            return np.clip(arr / 65535.0, 0, 1)
        if m > 255:
            return np.clip(arr / 4095.0, 0, 1)
        return np.clip(arr / 255.0, 0, 1)
    return np.clip(arr, 0, 1)
