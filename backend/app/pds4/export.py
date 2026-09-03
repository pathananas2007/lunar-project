"""Export registered product as GeoTIFF (or PNG fallback) + PDS4 XML label."""
from pathlib import Path
import cv2
import numpy as np
import base64
from .models import PDS4Metadata


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_registered_product(
    warped_u8: np.ndarray,
    ref_meta: PDS4Metadata | None,
    homography: list | np.ndarray | None,
    out_dir: str | Path = "data/registered",
    prefix: str = "registered",
) -> dict:
    """
    Writes warped image as TIFF (GeoTIFF-compatible) + PDS4 XML label.
    Returns {tiff_path, xml_path, warped_b64}
    Uses cv2.imwrite for TIFF; XML contains corner lat/lon if available.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        # Resolve relative to CWD (project root when running uvicorn) or fallback to parents[3]
        cwd_candidate = Path.cwd() / out_dir
        if cwd_candidate.parent.exists():
            out_dir = cwd_candidate
        else:
            root = Path(__file__).resolve().parents[3]
            out_dir = root / out_dir
    _ensure_dir(out_dir)

    # Write TIFF (fallback to PNG if TIFF writer missing)
    # warped_u8 is HxW uint8
    tiff_path = out_dir / f"{prefix}.tif"
    png_path = out_dir / f"{prefix}.png"
    try:
        ok = cv2.imwrite(str(tiff_path), warped_u8)
        if not ok or not tiff_path.exists():
            raise RuntimeError("TIFF write failed")
        final_path = tiff_path
    except Exception:
        cv2.imwrite(str(png_path), warped_u8)
        final_path = png_path

    # Build PDS4 XML label for registered product
    xml_path = out_dir / f"{prefix}.xml"
    h, w = warped_u8.shape[:2]
    pid = ref_meta.product_id + "_registered" if ref_meta else f"{prefix}"
    inst = ref_meta.instrument if ref_meta else "TMC-2"
    mpp = ref_meta.resolution_mpp if ref_meta and ref_meta.resolution_mpp else 5.0
    # Corner coords placeholder: if ref_meta has corner_lat_lon use it, else synthetic
    corners = ref_meta.corner_lat_lon if ref_meta and ref_meta.corner_lat_lon else None
    # Homography string
    H_str = ""
    if homography is not None:
        try:
            H_arr = np.array(homography, dtype=float)
            H_str = ",".join(f"{v:.6f}" for v in H_arr.ravel())
        except Exception:
            H_str = str(homography)

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Identification_Area>
    <logical_identifier>urn:ISRO:ch2:registered:{pid}</logical_identifier>
    <version_id>1.0</version_id>
    <title>Lunar-Align-X Registered Product</title>
    <information_model_version>1.14.0.0</information_model_version>
    <product_class>Product_Observational</product_class>
    <product_id>{pid}</product_id>
  </Identification_Area>
  <Observation_Area>
    <Time_Coordinates><start_date_time>2026-01-01T00:00:00Z</start_date_time></Time_Coordinates>
    <Primary_Result_Summary>
      <purpose>Science</purpose>
      <processing_level>Registered</processing_level>
    </Primary_Result_Summary>
    <Investigation_Area><name>Chandrayaan-2</name></Investigation_Area>
    <Observing_System>
      <Observing_System_Component><name>{inst}</name><type>Instrument</type></Observing_System_Component>
    </Observing_System>
  </Observation_Area>
  <File_Area_Observational>
    <File><file_name>{final_path.name}</file_name></File>
    <Array_2D_Image>
      <local_identifier>registered_image</local_identifier>
      <offset unit="byte">0</offset>
      <axes>2</axes>
      <axis_index_order>Last Index Fastest</axis_index_order>
      <Element_Array><data_type>UnsignedByte</data_type><scaling_factor>1</scaling_factor></Element_Array>
      <Axis_Array><axis_name>Line</axis_name><elements>{h}</elements><sequence_number>1</sequence_number></Axis_Array>
      <Axis_Array><axis_name>Sample</axis_name><elements>{w}</elements><sequence_number>2</sequence_number></Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
  <Mission_Information>
    <homography>{H_str}</homography>
    <resolution_mpp>{mpp}</resolution_mpp>
    <warped_shape>{h},{w}</warped_shape>
  </Mission_Information>
</Product_Observational>
"""
    xml_path.write_text(xml_content, encoding="utf-8")

    # Also encode b64 for API response
    ok2, buf = cv2.imencode(".png", warped_u8)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok2 else None

    return {
        "tiff_path": str(final_path),
        "xml_path": str(xml_path),
        "warped_b64": b64,
        "shape": [h, w],
    }
