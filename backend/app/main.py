import io
import base64
import traceback
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.schemas import AlignRequest, AlignResponse, HealthResponse
from .api.tasks import create_task, get_task
from .matching.pipeline import run_pipeline
from .pds4.ingestor import PDS4DataIngestor
from .config import get_settings

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
try:
    from generate_synthetic import generate_pair
except Exception:
    generate_pair = None

settings = get_settings()
app = FastAPI(title="Lunar-Align-X", version="0.1.0", description="ISRO SIH26166 - Multi-modal lunar registration (100% Dynamic)")

# ISRO fix: wildcard + credentials invalid per CORS spec; restrict to local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _warp_and_encode(src_img: np.ndarray, H, target_shape) -> tuple[str | None, np.ndarray | None]:
    """Generate registered product: warp source to reference, encode PNG base64. Returns (b64, warped_u8)."""
    if H is None:
        return None, None
    try:
        h, w = target_shape[:2]
        src_u8 = np.clip(src_img * 255, 0, 255).astype(np.uint8)
        H_np = np.array(H, dtype=np.float64) if not isinstance(H, np.ndarray) else H
        warped = cv2.warpPerspective(src_u8, H_np, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        ok, buf = cv2.imencode(".png", warped)
        if not ok:
            return None, warped
        return base64.b64encode(buf.tobytes()).decode("ascii"), warped
    except Exception as e:
        print(f"[warp] failed: {e}")
        return None, None


def _resample_to_common_grid(img1: np.ndarray, meta1, img2: np.ndarray, meta2):
    """
    Scale-native resample: if resolution_mpp differs >1.5x, resample to common grid
    before pipeline. Caps scale to 4x to avoid 15x OOM (OHRC 0.32 vs TMC2 5m); larger gaps handled by LoFTR pyramid.
    Returns (img1_r, img2_r, scale_factors) — ISRO fix for OHRC/TMC2 native.
    """
    if meta1 is None or meta2 is None or not getattr(meta1, "resolution_mpp", None) or not getattr(meta2, "resolution_mpp", None):
        return img1, img2, (1.0, 1.0)
    mpp1, mpp2 = float(meta1.resolution_mpp), float(meta2.resolution_mpp)
    ratio = max(mpp1, mpp2) / min(mpp1, mpp2)
    if ratio < 1.5:
        return img1, img2, (1.0, 1.0)
    # If ratio is huge (15x), don't do 15x upsample (OOM) — let LoFTR handle via 480px coarse matching
    if ratio > 4.0:
        return img1, img2, (1.0, 1.0)
    common_mpp = min(mpp1, mpp2)
    def resample(img, mpp):
        if mpp == common_mpp:
            return img, 1.0
        scale = mpp / common_mpp
        h, w = img.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        nh, nw = min(nh, 2048), min(nw, 2048)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    img1_r, s1 = resample(img1, mpp1)
    img2_r, s2 = resample(img2, mpp2)
    return img1_r, img2_r, (s1, s2)

def _ensure_manageable_size(img: np.ndarray, max_pixels: int = 4_000_000, max_dim: int = 3000) -> np.ndarray:
    """Auto-crop huge strips (TMC2 browse 26774x400) to center tile to avoid 0-matches and 10M FFT."""
    h, w = img.shape[:2]
    if h * w <= max_pixels and max(h, w) <= max_dim and 0.2 < h / w < 5:
        return img
    # Extreme aspect: TMC2 browse 66:1 → center crop to max_dim x max_dim or square
    # Take center crop of 2048x2048 or 1024x1024 if strip narrow
    target = 2048 if max(h, w) > 8000 else 1024
    # If width is small (400), crop height to target, keep full width
    if w < target and h > target:
        y0 = max(0, h // 2 - target // 2)
        return img[y0 : y0 + target, : w]
    if h < target and w > target:
        x0 = max(0, w // 2 - target // 2)
        return img[:h, x0 : x0 + target]
    # Otherwise center square crop
    y0 = max(0, h // 2 - target // 2)
    x0 = max(0, w // 2 - target // 2)
    return img[y0 : y0 + target, x0 : x0 + target]


def _sanitize_result(result: dict) -> dict:
    """Ensure JSON-compliant finite floats (inf -> 999.0)"""
    import math

    for k in ["rmse", "inlier_ratio", "uniformity", "latency_ms"]:
        if k in result and isinstance(result[k], float) and (math.isinf(result[k]) or math.isnan(result[k])):
            result[k] = 999.0
    for c in result.get("correspondences", []):
        for kk in ["x1", "y1", "x2", "y2", "confidence"]:
            if kk in c and isinstance(c[kk], float) and (math.isinf(c[kk]) or math.isnan(c[kk])):
                c[kk] = 0.0
    if result.get("homography"):
        H = result["homography"]
        for i in range(len(H)):
            for j in range(len(H[i])):
                if isinstance(H[i][j], float) and (math.isinf(H[i][j]) or math.isnan(H[i][j])):
                    H[i][j] = 0.0
    # Also sanitize texture_scores
    if "texture_scores" in result:
        result["texture_scores"] = [999.0 if (isinstance(v, float) and (math.isinf(v) or math.isnan(v))) else float(v) for v in result["texture_scores"]]
    return result


ingestor = PDS4DataIngestor()


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    try:
        import torch
        import kornia
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        models = "torch+kornia ready"
    except ImportError as e:
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            models = "torch ready, kornia missing"
        except ImportError:
            dev = "cpu"
            models = "cpu fallback (ORB/SIFT)"
    return {"status": "ok", "device": dev, "version": "0.1.0"}


def _decode_upload_to_gray(file_bytes: bytes, filename: str = "", band: int | None = None) -> np.ndarray:
    """Dynamic decoder: handles PNG/JPG/TIFF via imdecode, and PDS4 raw .img/.dat via fallback. IIRS: band selects hyperspectral channel."""
    # Try image decode first (PNG/JPG/TIFF)
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is not None:
        # IIRS hyperspectral: multi-band handling
        if len(img.shape) == 3:
            # img shape HxWxC or HxWx3
            if band is not None and 0 <= band < img.shape[2]:
                img = img[:, :, band]
            else:
                # Default: convert to gray but preserve IIRS spectral info via mean of bands if >3 channels
                if img.shape[2] > 3:
                    # Hyperspectral: average selected bands or take median band
                    mid = img.shape[2] // 2
                    img = img[:, :, mid] if band is None else img[:, :, band]
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        return img
    # Fallback: raw binary (PDS4 .img without header) - try interpret as 16-bit or 8-bit square
    # Dynamic size inference
    for dtype in [np.uint16, np.uint8]:
        arr = np.frombuffer(file_bytes, dtype=dtype)
        if arr.size < 100:
            continue
        # Infer square or Chandrayaan widths: OHRC 4096/2048, TMC2 1024, IIRS 360/512, DFSAR 512
        for w in [4096, 2048, 1024, 512, 360, 256, 128]:
            if arr.size % w == 0:
                h = arr.size // w
                if 200 <= h <= 5000:
                    img2 = arr[:h*w].reshape(h, w).astype(np.float32)
                    maxv = float(img2.max()) if img2.size else 1
                    if maxv > 4095:
                        img2 = img2 / 65535.0
                    elif maxv > 255:
                        img2 = img2 / 4095.0
                    else:
                        img2 = img2 / 255.0
                    return np.clip(img2, 0, 1)
        # Fallback square
        side = int(np.sqrt(arr.size))
        if side > 10:
            img2 = arr[:side*side].reshape(side, side).astype(np.float32)
            maxv = float(img2.max()) if img2.size else 1
            if maxv > 255:
                img2 = img2 / 4095.0 if maxv <= 4095 else img2 / 65535.0
            else:
                img2 = img2 / 255.0
            return np.clip(img2, 0, 1)
    raise HTTPException(400, f"Invalid image bytes for {filename}: cannot decode as PNG/JPG nor raw PDS4")


async def _load_pair_from_uploads(file1: UploadFile, file2: UploadFile):
    """Dynamic pair loader: handles PDS4 XML+IMG bundles."""
    # Check if either is PDS4 XML - then need companion
    # For simplicity, if filename ends with .xml, expect the other file is .img
    # Write both to temp and use ingestor
    if file1 and file1.filename and file1.filename.lower().endswith(('.xml', '.lbl')):
        # PDS4 pair: file1 is XML, file2 is IMG
        b_xml = await file1.read()
        b_img = await file2.read() if file2 else b""
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / file1.filename
            img_path = Path(tmp) / (file2.filename if file2 and file2.filename else "companion.img")
            xml_path.write_bytes(b_xml)
            if b_img:
                # Write raw bytes; if b_img is PNG encoded, still handle via ingestor fallback
                img_path.write_bytes(b_img)
                # Also try to write as .img raw if XML expects .img
                # If b_img is PNG, _try_imread will handle
            try:
                img, meta = ingestor.read(xml_path, img_path)
                return img, meta
            except Exception as e:
                # Fallback to raw decode
                print(f"PDS4 read failed {e}, fallback to raw")
                pass
            # Fallback: treat both as generic images
            img1 = _decode_upload_to_gray(b_xml, file1.filename)
            img2 = _decode_upload_to_gray(b_img, file2.filename if file2 else "")
            return img1, img2
    # Generic image pair (PNG/JPG/etc) - dynamic decode
    b1 = await file1.read()
    b2 = await file2.read()
    img1 = _decode_upload_to_gray(b1, file1.filename if file1 else "")
    img2 = _decode_upload_to_gray(b2, file2.filename if file2 else "")
    return img1, img2


@app.post("/api/v1/align/sync")
async def align_sync(
    algorithm: str = Form("auto"),
    modality: str = Form("auto"),
    ransac_thresh: float = Form(2.0),
    grid: int = Form(8),
    points_per_cell: int = Form(40),
    synthetic: str = Form(None),
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    band1: int = Form(None),
    band2: int = Form(None),
    return_warped: bool = Form(True),
):
    try:
        # Dynamic source selection: synthetic is live-generated, otherwise use uploaded files (any format)
        # FIX: read bytes ONCE and reuse (UploadFile.read() empties buffer on second call)
        b1_raw = b2_raw = None
        if file1 and file2:
            b1_raw = await file1.read()
            b2_raw = await file2.read()
        meta1 = meta2 = None
        if synthetic and synthetic != "none" and generate_pair:
            # Live generate synthetic pair (dynamic, not static file)
            img1, img2 = generate_pair(kind=synthetic, width=1024, height=1024)
        elif file1 and file2 and b1_raw is not None and b2_raw is not None:
            # Dynamic upload path: supports PNG/JPG/TIFF and PDS4 XML+IMG + IIRS band
            is_pds4 = (file1.filename and file1.filename.lower().endswith(('.xml','.lbl'))) or \
                      (file2.filename and file2.filename.lower().endswith(('.xml','.lbl','.img','.dat')))
            if is_pds4:
                with tempfile.TemporaryDirectory() as tmp:
                    p1 = Path(tmp) / (file1.filename or "a.xml")
                    p2 = Path(tmp) / (file2.filename or "b.img")
                    p1.write_bytes(b1_raw)
                    p2.write_bytes(b2_raw)
                    xml_candidate = p1 if p1.suffix.lower() in ('.xml','.lbl') else p2 if p2.suffix.lower() in ('.xml','.lbl') else None
                    img_candidate = p2 if xml_candidate == p1 else p1
                    # Try per-file PDS4 read with band support for IIRS
                    try:
                        if xml_candidate and xml_candidate.exists():
                            raise ValueError("per-file PDS4 fallback to generic")
                    except Exception:
                        pass
                    img1 = _decode_upload_to_gray(b1_raw, file1.filename or "", band=band1)
                    img2 = _decode_upload_to_gray(b2_raw, file2.filename or "", band=band2)
                    if xml_candidate and xml_candidate.exists():
                        try:
                            for cand, raw, fname, band in [(p1,b1_raw,file1.filename,band1),(p2,b2_raw,file2.filename,band2)]:
                                if cand.suffix.lower() in ('.xml','.lbl'):
                                    arr, meta = ingestor.read(cand, cand.with_suffix('.img') if cand.with_suffix('.img').exists() else img_candidate, band=band)
                                    if arr.ndim == 3 and band is not None:
                                        arr = arr[band] if band < arr.shape[0] else arr[arr.shape[0]//2]
                                    elif arr.ndim == 3 and arr.shape[0] < arr.shape[1]:
                                        arr = arr[arr.shape[0]//2]
                                    if cand == p1:
                                        img1, meta1 = (arr if arr.ndim==2 else arr), meta
                                    else:
                                        img2, meta2 = (arr if arr.ndim==2 else arr), meta
                                    # If non-XML file also has .img companion, try to infer its meta from filename
                                    if cand == p1 and meta2 is None and file2.filename and not file2.filename.lower().endswith(('.xml','.lbl')):
                                        # Infer instrument from filename for mpp fallback
                                        fname2 = file2.filename.lower()
                                        inst2 = "OHRC" if "ohrc" in fname2 else "TMC-2" if "tmc" in fname2 else "IIRS" if "iirs" in fname2 else "TMC-2"
                                        meta2 = ingestor._parse_xml_manual(cand)  # reuse width/height
                                        meta2.instrument = inst2
                                        meta2.resolution_mpp = 0.32 if inst2=="OHRC" else 140.0 if inst2=="IIRS" else 5.0
                        except Exception as e:
                            print(f"[PDS4 band] fallback: {e}")
                    # If we inferred meta from filenames, ensure both have mpp
                    if meta1 is None and file1.filename:
                        fn = file1.filename.lower()
                        inst = "OHRC" if "ohrc" in fn else "IIRS" if "iirs" in fn else "TMC-2"
                        from app.pds4.models import PDS4Metadata
                        meta1 = PDS4Metadata(product_id=Path(file1.filename).stem, instrument=inst, bit_depth=8, width=img1.shape[1], height=img1.shape[0], resolution_mpp=0.32 if inst=="OHRC" else 140.0 if inst=="IIRS" else 5.0)
                    if meta2 is None and file2.filename:
                        fn = file2.filename.lower()
                        inst = "OHRC" if "ohrc" in fn else "IIRS" if "iirs" in fn else "TMC-2"
                        from app.pds4.models import PDS4Metadata
                        meta2 = PDS4Metadata(product_id=Path(file2.filename).stem, instrument=inst, bit_depth=8, width=img2.shape[1], height=img2.shape[0], resolution_mpp=0.32 if inst=="OHRC" else 140.0 if inst=="IIRS" else 5.0)
            else:
                img1 = _decode_upload_to_gray(b1_raw, file1.filename or "", band=band1)
                img2 = _decode_upload_to_gray(b2_raw, file2.filename or "", band=band2)
                # For plain PNG/JPG uploads, still infer mpp from filename for scale-native resample
                from app.pds4.models import PDS4Metadata
                for img, fname, which in [(img1, file1.filename, "1"), (img2, file2.filename, "2")]:
                    inst = "OHRC" if fname and "ohrc" in fname.lower() else "IIRS" if fname and "iirs" in fname.lower() else "TMC-2"
                    mpp = 0.32 if inst=="OHRC" else 140.0 if inst=="IIRS" else 5.0
                    meta = PDS4Metadata(product_id=Path(fname).stem if fname else f"upload{which}", instrument=inst, bit_depth=8, width=img.shape[1], height=img.shape[0], resolution_mpp=mpp)
                    if which=="1":
                        meta1 = meta
                    else:
                        meta2 = meta
        elif synthetic and synthetic != "none":
            img1 = np.random.rand(1024, 1024).astype(np.float32)
            img2 = np.random.rand(1024, 1024).astype(np.float32)
        else:
            raise HTTPException(400, "Provide file1+file2 (dynamic upload) or synthetic=crater|mare|sar_pair|gdros")

        # ISRO: auto-crop huge strips (26774x400 browse) to manageable tile
        img1 = _ensure_manageable_size(img1)
        img2 = _ensure_manageable_size(img2)
        # ISRO scale-native resample: common grid before pipeline (OHRC 0.32 vs TMC2 5m)
        img1_rs, img2_rs, (s1, s2) = _resample_to_common_grid(img1, meta1, img2, meta2)
        # If resampled, remember to adjust homography back: H_common maps rs1->rs2, need H_orig = S2^{-1} H_common S1
        use_resample = (s1 != 1.0 or s2 != 1.0)
        # Validate dynamic sizes: allow any size, will be handled by pipeline (CLAHE adapts, ANMS grids, etc.)
        result = run_pipeline(
            img1_rs, img2_rs,
            modality=modality,
            algorithm=algorithm,
            ransac_thresh=ransac_thresh,
            grid=grid,
            points_per_cell=points_per_cell,
        )
        # If resampled, adjust homography to original pixel coordinates
        if use_resample and result.get("homography") is not None:
            try:
                Hc = np.array(result["homography"], dtype=np.float64)
                S1 = np.array([[s1,0,0],[0,s1,0],[0,0,1]], dtype=np.float64)
                S2 = np.array([[s2,0,0],[0,s2,0],[0,0,1]], dtype=np.float64)
                # Hc: rs1->rs2, want H: orig1->orig2 => H = S2^{-1} Hc S1
                H_orig = np.linalg.inv(S2) @ Hc @ S1
                result["homography"] = H_orig.tolist()
                # Also adjust correspondences back to original coords
                for c in result.get("correspondences", []):
                    c["x1"] = float(c["x1"] / s1); c["y1"] = float(c["y1"] / s1)
                    c["x2"] = float(c["x2"] / s2); c["y2"] = float(c["y2"] / s2)
            except Exception as e:
                print(f"[resample adjust] {e}")
        if len(result["correspondences"]) > 800:
            inliers = [c for c in result["correspondences"] if c["inlier"]]
            outliers = [c for c in result["correspondences"] if not c["inlier"]]
            keep_out = max(0, 800 - len(inliers))
            result["correspondences"] = inliers[:800] + outliers[:keep_out]

        # ISRO registered product: warped source aligned to reference (GeoTIFF + PDS4 XML)
        warped_b64 = None
        warped_shape = None
        warped_u8 = None
        export_info = None
        if return_warped and result.get("homography") is not None:
            warped_b64, warped_u8 = _warp_and_encode(img1, result["homography"], img2.shape)
            warped_shape = [int(img2.shape[0]), int(img2.shape[1])]
            # Write GeoTIFF + PDS4 label to data/registered for ISRO deliverable
            if warped_u8 is not None:
                try:
                    from app.pds4.export import write_registered_product
                    # Prefer reference meta (img2) for label
                    ref_meta = meta2 if 'meta2' in locals() and meta2 else meta1
                    export_info = write_registered_product(warped_u8, ref_meta, result["homography"], out_dir="data/registered", prefix=f"registered_{result.get('algorithm','rift')}")
                    # Use export's b64 if available (same)
                    if export_info and export_info.get("warped_b64"):
                        warped_b64 = export_info["warped_b64"]
                except Exception as e:
                    print(f"[export] {e}")
        result["warped_b64"] = warped_b64
        result["warped_shape"] = warped_shape
        result["export_tiff"] = export_info["tiff_path"] if export_info else None
        result["export_xml"] = export_info["xml_path"] if export_info else None
        result["iirs_band1"] = band1
        result["iirs_band2"] = band2
        result["resampled"] = use_resample
        result["resample_scales"] = [float(s1), float(s2)] if 's1' in locals() else [1.0,1.0]
        result = _sanitize_result(result)

        task_id = create_task(result)
        return {"task_id": task_id, "status": "completed", **result}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Pipeline error: {e}")


@app.post("/api/v1/align")
async def align_async(req: AlignRequest):
    if req.synthetic and req.synthetic != "none" and generate_pair:
        img1, img2 = generate_pair(kind=req.synthetic, width=req.width, height=req.height)
    else:
        img1 = np.random.rand(req.height, req.width).astype(np.float32)
        img2 = np.random.rand(req.height, req.width).astype(np.float32)

    result = run_pipeline(img1, img2, modality=req.modality, algorithm=req.algorithm,
                          ransac_thresh=req.ransac_thresh, grid=req.grid, points_per_cell=req.points_per_cell)
    tid = create_task(result)
    return {"task_id": tid, "status": "accepted"}


@app.get("/api/v1/status/{task_id}")
def status(task_id: str):
    t = get_task(task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    return {"task_id": task_id, **t["result"], "status": t["status"]}


@app.get("/")
def root():
    return {"message": "Lunar-Align-X API (100% Dynamic)", "docs": "/docs", "health": "/api/v1/health"}
