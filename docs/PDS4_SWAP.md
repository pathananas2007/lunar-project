# PDS4 Real-Data Swap Guide

MVP currently uses `scripts/generate_synthetic.py` to generate mock `float32 [0,1]` lunar terrain and `data/samples/*.xml` stubs. To swap to real ISSDC PRADAN Chandrayaan-2 data:

1. **Download via PRADAN portal** (requires registration):
   ```bash
   python scripts/download_pradan.py --url https://pradan.issdc.gov.in/ch2/TMC2/...zip --out data/raw/ch2_tmc2.zip
   # Handles Jump-To-File resume for ZIP volume limits, see script for Range header logic
   unzip data/raw/ch2_tmc2.zip -d data/raw/
   ```

2. **PDS4 ingestion** is already production-ready in `backend/app/pds4/ingestor.py:1`:
   ```python
   from app.pds4.ingestor import PDS4DataIngestor
   ingestor = PDS4DataIngestor()
   img, meta = ingestor.read("data/raw/CH2_TMC2_....xml") # auto-finds .img via label
   # img is float32 [0,1] normalized, meta has footprint lat/lon, scaling, bit_depth
   # Supports pds4_tools.read() + manual Fallback via xml.etree + np.fromfile memmap
   ```

3. **Bit-depth handling**: raw 12/16-bit `uint16` arrays are auto-scaled to `float32/4095` or `/65535` in `_normalize()`. No code change needed.

4. **API swap**: `POST /api/v1/align/sync` already accepts multipart file uploads (PNG/JPG) and raw binary via ingestor. For real PDS4, upload the `.xml` + `.img` pair as `file1`/`file2` or mount `data/raw` and call ingestor directly in `app/main.py:1`.

5. **Test with real label**: copy a real `.xml` into `data/samples/` and run:
   ```bash
   cd backend; uv run pytest tests/test_pds4_ingestor.py -v
   ```

No pipeline changes required; CLAHE, RIFT, ANMS, LoFTR router, cornerSubPix, RANSAC all operate on `float32 [0,1]` abstraction.
