from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    device: str = "cpu"  # cpu | cuda
    ransac_thresh: float = 3.0
    anms_grid: int = 8
    anms_points_per_cell: int = 40
    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
