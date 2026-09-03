from pydantic import BaseModel
from typing import Optional, List, Any


class AlignRequest(BaseModel):
    algorithm: str = "auto"  # auto | rift | loftr | lightglue
    modality: str = "auto"   # auto | optical | sar | infrared
    ransac_thresh: float = 3.0
    grid: int = 8
    points_per_cell: int = 40
    # For demo: allow base64 or synthetic size
    synthetic: Optional[str] = None  # "crater" | "mare" | "sar_pair"
    width: int = 1024
    height: int = 1024
    band1: Optional[int] = None  # IIRS hyperspectral band index
    band2: Optional[int] = None
    return_warped: bool = True


class Correspondence(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    inlier: bool
    confidence: float


class AlignResponse(BaseModel):
    task_id: str
    status: str
    algorithm: str
    total_keypoints: int
    inlier_count: int
    inlier_ratio: float
    rmse: float
    homography: Optional[List[List[float]]]
    correspondences: List[Any]
    uniformity: float
    latency_ms: float
    warped_b64: Optional[str] = None
    warped_shape: Optional[List[int]] = None
    iirs_band1: Optional[int] = None
    iirs_band2: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    version: str
