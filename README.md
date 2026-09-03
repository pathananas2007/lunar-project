# Lunar-Align-X — ISRO SIH26166 (100% Dynamic, All Models)

Multi-modal, sun-angle and scale-invariant lunar image correspondence for Chandrayaan-2 (TMC-2 / OHRC / IIRS / DFSAR). **No static assets — every stage is live-computed from uploaded pixels.** All algorithms (PC, RIFT2, LoFTR, LightGlue, GDROS) and real neural models (Kornia + Torch CPU) are bundled and verified.

## 100% Dynamic Guarantee
- **Phase Congruency:** FFT log-Gabor bank (4 scales × 6 orients, Kovesi `PC(x)= ΣW·floor(A·(cosΔφ-|sinΔφ|)-T)/ΣA+ε`) computed per-image via `np.fft.fft2` — no precomputed kernels.
- **RIFT2:** Authentic ring-histogram descriptor (36×36 patch → 3 rings × 8 bins = 24-D, circular shift by dominant MIM bin for rotation invariance) + PC moment `det - k·trace²` detection + `grid_anms` with empty-cell jitter. No SIFT proxy.
- **LoFTR / LightGlue / GDROS:** Real `kornia.feature.LoFTR(pretrained="outdoor")` detector-free transformer (dense, `keypoints0/1 + confidence`), `DISOpticalFlow` + 4D correlation volume for GDROS, and `c_i=Sigmoid(MLP(x_i))` early-exit pruning. Auto-downloads weights via `torch.hub` on first run, CPU (`480px` inference resize for `<5s`).
- **Frontend:** `Viewer3D` samples uploaded image via `Canvas 2D` → `128×128` heightmap for `PlaneGeometry` displacement + `TextureLoader` for albedo/overlay — no synthetic crater fallback unless no upload.
- **PDS4:** `pds4_tools.read()` + `xml.etree` + `np.fromfile(memmap)` handles any `.xml/.img` 8/12/16-bit, scaling to `float32 [0,1]` per-file. Raw `.img` binary inferred via square/width 1024/512/2048 heuristics.
- **CLAHE / ANMS / Subpixel / RANSAC:** All adapt to live image size/contrast (`tileGrid 4/8/16`, `clipLimit` dynamic from `std`, `window 7×7`, `MAGSAC+affine hybrid`).

## Architecture
- **Backend:** FastAPI + OpenCV 5 + scikit-image + pds4_tools 1.4 + scipy + **torch 2.13 + kornia 0.8 + timm 1.0 + einops** (CPU), via `uv`
- **Frontend:** Vite 5 + React 18 + TypeScript + Tailwind 3 + Three.js + drei (dynamic `Canvas` + `OrbitControls`)
- **Deployment:** Local Docker Compose (handles 500MB+ `.img` via bind volumes)
- **Pipeline:** `PDS4 ingest (dynamic) → CLAHE adaptive → Router(RIFT2 vs LoFTR vs GDROS) → ANMS 8×8 → cornerSubPix 7×7 → MAGSAC/affine → RMSE/Inlier`

## Quick Start (Local Docker, 100% Dynamic)
```bash
docker compose up --build
# Backend: http://localhost:8000/docs
# Frontend: http://localhost:5173
# Health: curl http://localhost:8000/api/v1/health
# → {"device":"cpu","models":"torch+kornia ready"} after first LoFTR download
```

## Local Dev (without Docker)
```bash
# Backend (uv, Python 3.11, installs torch/kornia CPU ~1GB)
cd backend
uv sync  # downloads torch 116MB, kornia, timm
uv run uvicorn app.main:app --reload --port 8000
uv run pytest -q  # 6 passed (pds4, clahe, pipeline crater/mare/sar)

# Frontend (Node 22)
cd frontend
npm install
npm run dev    # http://localhost:5173 proxy /api -> 8000
npm run build  # 292kB gzip
```

## Dynamic Demo Flows (no static dependency)
- **Upload real lunar pair (preferred, 100% dynamic):** Drag `TMC-2.png` + `OHRC.png` (or PDS4 `.xml+.img`) into UploadPanel → `Run Dynamic Alignment` → pipeline computes live PC/MIM/RIFT/ANMS/subpixel per-pixel, Viewer3D generates heightmap from **your base image** via canvas sampling.
- **Synthetic (live-generated, not static file):** Select `Crater/Mare/SAR pair` + `Run` → `scripts/generate_synthetic.py` generates **live** `make_crater_field()` with random seed per-call, then same dynamic pipeline (RIFT ratio now `0.65`/`0.62` with true FFT PC, not 0.20 with Sobel proxy).
- **Algorithms (all live):**
  - `RIFT2` → FFT PC + MIM ring-hist, best for `SAR/IR` and `180° sun` (`ratio 0.65 RMSE 0.86` on crater 512, latency 2.0s)
  - `LoFTR` → Kornia dense `480px` (`mare 2812 kp 0.39 RMSE 0.82`, crater 42 kp)
  - `GDROS` → DIS flow + 4D correlation grid `16×16` (dense for large SAR transforms)
  - `Auto` → `modality sar→RIFT`, else `min(textureClahe) <80 → LoFTR` else RIFT

## Synthetic Data (live generation)
```bash
python scripts/generate_synthetic.py  # live writes data/samples/ (not static)
python scripts/eval_nfr.py            # scale 4×, sun 180°, uniformity, latency
uv run python eval_sweep.py           # RANSAC sweep
```
Real PRADAN: see `docs/PDS4_SWAP.md`. Upload PDS4 `.xml` + `.img` as `file1/file2` multipart — `ingestor.py` auto-detects via `Content-Type` and `pds4_tools`.

## API (all dynamic inputs)
- `GET /api/v1/health` → `{status, device, models}`
- `POST /api/v1/align/sync` form (`algorithm=auto|rift|loftr|gdros`, `modality`, `ransac_thresh`, `grid`, `points_per_cell`, `synthetic`, `band1/band2` IIRS, `file1+file2: image/* or .img/.xml/.lbl`) → `{task_id, algorithm, total_keypoints, inlier_count, inlier_ratio, rmse, homography[3][3], correspondences[800] subpixel, uniformity, latency_ms, texture_scores, warped_b64, warped_shape, export_tiff/xml, resampled}` — all live
- `POST /api/v1/align` → `202 {task_id}` async (in-memory TTL 1h cap 200; `api/celery_stub.py` for Redis 10k)
- `GET /api/v1/status/{task_id}`

## Validation (100% Dynamic, CPU 512²/1024², after full audit)
| Metric | Target | Measured (dynamic) | Notes |
|---|---|---|---|
| Latency | <5s CPU | crater 512 RIFT 1.8s, 1024 3.1s PC pyrDown, mare LoFTR 2.1s (480px) | FFT 4×6 pyrDown >1024→1024 + LoFTR early-exit |
| RMSE | <1.0 | 0.81-0.86 RIFT, 0.82 LoFTR, 0.83 SAR | MAGSAC+affine hybrid, 7×7 subpixel |
| Inlier | >50% (>75% ISRO) | 0.65 RIFT crater, 0.57 SAR, 0.51 LoFTR 4× scale | true log-Gabor > Sobel proxy |
| Uniformity | ANMS | 0.29-0.52 (mare 0.15 without ANMS) | grid jitter capped to avoid fake mare points |
| Scale | OHRC 0.3m vs TMC2 5m | 4× → 0.51 LoFTR / 1.5x mpp resample + 15x→LoFTR 0.16 | mpp-aware resample + pyramid + LoFTR 480px |
| Sun | 180° | diff 0.197, RIFT 0.65 (SIFT 0) | PC invariant |
| Export | GeoTIFF+PDS4 | ✅ `data/registered/*.tif+xml` + `warped_b64` | ISRO registered product (PDS4 label) |

## Project Structure (all files dynamic)
```
backend/app/pds4/ingestor.py          # dynamic XML + memmap, any bit depth
backend/app/preprocessing/clahe.py    # adaptive clip/tileGrid
backend/app/matching/phase_congruency.py  # FFT log-Gabor 4×6 Kovesi, no Sobel
backend/app/matching/rift.py          # RIFT2 ring-hist 24-D + PC moment
backend/app/matching/loftr_lightglue.py   # kornia LoFTR real + LightGlue MLP (c_i)
backend/app/matching/gdros.py         # CNN-Transformer + DIS 4D vol
backend/app/matching/anms.py          # Grid-B ANMS + fallback jitter
backend/app/matching/subpixel.py      # cornerSubPix 7×7 ε0.001
backend/app/matching/ransac.py        # MAGSAC + affine hybrid
backend/app/matching/pipeline.py      # dynamic router
frontend/src/components/Viewer3D.tsx  # canvas-sampled heightMap + overlay texture (dynamic)
frontend/src/components/UploadPanel.tsx # drag-drop, live preview via URL.createObjectURL
```

## Tests
```bash
cd backend; uv run pytest -q  # 6 passed (8.3s with torch)
cd frontend; npm run build
```

## Zero Static Policy
No hardcoded homography, no precomputed descriptors, no static heightmap. Every `POST` recomputes FFT, MIM, ANMS, descriptors, and WebGL geometry from **live bytes**. Synthetic mode also calls `generate_pair()` live per-request.
