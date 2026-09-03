"""PDS4DataIngestor - parses PDS4 XML + binary arrays into float32 tensors.
Falls back to synthetic raw arrays if XML missing. Handles memmap for large .img.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

try:
    from pds4_tools import pds4_read
    HAS_PDS4 = True
except ImportError:
    HAS_PDS4 = False

from .models import PDS4Metadata


class PDS4DataIngestor:
    """Ingests PDS4 .xml label + .img/.dat binary into normalized float32."""

    PDS4_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}

    def read(self, xml_path: str | Path, img_path: Optional[str | Path] = None, band: Optional[int] = None) -> Tuple[np.ndarray, PDS4Metadata]:
        xml_path = Path(xml_path)
        if img_path is None:
            # Infer .img from label
            img_path = xml_path.with_suffix(".img")
            if not img_path.exists():
                img_path = xml_path.with_suffix(".dat")

        # Try pds4_tools first
        if HAS_PDS4 and xml_path.exists():
            try:
                struct = pds4_read(str(xml_path), quiet=True)
                arr = self._extract_array_from_struct(struct)
                # IIRS cube: select band if requested (arr may be HxWxB or BxHxW)
                if band is not None and arr.ndim == 3:
                    if arr.shape[0] < arr.shape[1] and arr.shape[0] < 100:  # BxHxW
                        arr = arr[band] if 0 <= band < arr.shape[0] else arr[arr.shape[0]//2]
                    elif arr.shape[2] < 100:  # HxWxB
                        arr = arr[:, :, band] if 0 <= band < arr.shape[2] else arr[:, :, arr.shape[2]//2]
                elif band is None and arr.ndim == 3:
                    # Default: middle band for IIRS
                    if arr.shape[0] < 100:
                        arr = arr[arr.shape[0]//2]
                    elif arr.shape[2] < 100:
                        arr = arr[:, :, arr.shape[2]//2]
                meta = self._extract_meta_from_struct(struct, xml_path)
                img_f = self._normalize(arr)
                # Ensure 2D after band selection
                if img_f.ndim == 3:
                    img_f = img_f[img_f.shape[0]//2] if img_f.shape[0] < 50 else img_f[:, :, img_f.shape[2]//2]
                return img_f, meta
            except Exception as e:
                print(f"[PDS4] pds4_tools failed: {e}, falling back to manual parse")

        # Manual fallback: parse XML + read raw binary
        if xml_path.exists():
            meta = self._parse_xml_manual(xml_path)
            if Path(img_path).exists():
                arr = self._read_raw_binary(img_path, meta)
                if band is not None and arr.ndim == 3:
                    if arr.shape[0] < 100:
                        arr = arr[band] if 0 <= band < arr.shape[0] else arr[arr.shape[0]//2]
                    else:
                        arr = arr[:, :, band] if 0 <= band < arr.shape[2] else arr[:, :, arr.shape[2]//2]
                elif arr.ndim == 3:
                    if arr.shape[0] < 100:
                        arr = arr[arr.shape[0]//2]
                    elif arr.shape[2] < 100:
                        arr = arr[:, :, arr.shape[2]//2]
                return self._normalize(arr), meta

        # Synthetic fallback: if we receive direct numpy/array path (for tests/demo)
        if Path(img_path).exists():
            arr = np.load(str(img_path)) if str(img_path).endswith(".npy") else self._try_imread(img_path)
            meta = PDS4Metadata(product_id=Path(img_path).stem, instrument="TMC-2", bit_depth=8, width=arr.shape[1], height=arr.shape[0])
            return self._normalize(arr), meta

        raise FileNotFoundError(f"No valid PDS4 label or image found: {xml_path}, {img_path}")

    def read_array(self, array_or_path) -> Tuple[np.ndarray, PDS4Metadata]:
        """Accept numpy array directly (synthetic terrain path)."""
        if isinstance(array_or_path, np.ndarray):
            arr = array_or_path
            meta = PDS4Metadata(product_id="synthetic", instrument="TMC-2", bit_depth=8, width=arr.shape[1], height=arr.shape[0])
            return self._normalize(arr), meta
        if isinstance(array_or_path, (str, Path)) and Path(array_or_path).exists():
            p = Path(array_or_path)
            if p.suffix == ".npy":
                return self.read_array(np.load(str(p)))
            return self.read(p)
        raise ValueError("Unsupported input for read_array")

    # ---- helpers ----
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        # Detect bit depth by max value
        maxv = float(arr.max()) if arr.size else 1.0
        if maxv > 1.5:  # not yet normalized
            if maxv > 4095:
                arr = arr / 65535.0
            elif maxv > 255:
                arr = arr / 4095.0
            else:
                arr = arr / 255.0
        return np.clip(arr, 0, 1)

    def _extract_array_from_struct(self, struct) -> np.ndarray:
        # pds4_tools returns OrderedDict-like with .__getitem__
        for key in struct.keys():
            obj = struct[key]
            if hasattr(obj, "data"):
                return np.array(obj.data)
            if isinstance(obj, np.ndarray):
                return obj
        # fallback: first array-like
        first = list(struct.values())[0]
        if hasattr(first, "data"):
            return np.array(first.data)
        return np.array(first)

    def _extract_meta_from_struct(self, struct, xml_path: Path) -> PDS4Metadata:
        # Best-effort extract from struct LABEL — infer instrument/bands/mpp
        pid = xml_path.stem
        inst = "TMC-2"
        bands = None
        mpp = None
        try:
            for k in struct.keys():
                lk = k.lower()
                if "iirs" in lk:
                    inst = "IIRS"; mpp = 140.0; break
                if "ohrc" in lk:
                    inst = "OHRC"; mpp = 0.32; break
                if "dfsar" in lk:
                    inst = "DFSAR"; mpp = 25.0; break
        except Exception:
            pass
        if inst == "OHRC":
            mpp = 0.32
        elif inst == "TMC-2":
            mpp = 5.0
        return PDS4Metadata(product_id=pid, instrument=inst, bit_depth=16, width=1024, height=1024, bands=bands, resolution_mpp=mpp)

    def _parse_xml_manual(self, xml_path: Path) -> PDS4Metadata:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        ns = self.PDS4_NS
        def find_text(xpath):
            el = root.find(xpath, ns)
            return el.text.strip() if el is not None and el.text else None

        pid = find_text(".//pds:product_id") or xml_path.stem
        inst = find_text(".//pds:instrument_name") or find_text(".//pds:Instrument_Name") or "TMC-2"
        # Normalize instrument label
        if "IIRS" in inst.upper():
            inst = "IIRS"
        elif "OHRC" in inst.upper():
            inst = "OHRC"
        elif "DFSAR" in inst.upper():
            inst = "DFSAR"
        # Determine dtype, width/height from Array_2D_Image / Array_3D_Image (IIRS cube)
        w = find_text(".//pds:elements") or find_text(".//pds:line_samples") or "1024"
        h = find_text(".//pds:lines") or "1024"
        bands = find_text(".//pds:bands") or find_text(".//pds:Band_Bin_Number")
        # Resolution mpp from various tags
        mpp_txt = (
            find_text(".//pds:pixel_scale") or find_text(".//pds:ground_sample_distance")
            or find_text(".//pds:Instrument_Resolution") or find_text(".//pds:spatial_resolution")
        )
        try:
            w, h = int(w), int(h)
        except Exception:
            w, h = 1024, 1024
        try:
            bands = int(bands) if bands else None
        except Exception:
            bands = None
        # Parse mpp: ISRO uses 0.3 for OHRC, 5.0 for TMC2, ~140 for IIRS
        mpp = None
        if mpp_txt:
            try:
                # May contain "5.0 m/pixel" -> extract float
                import re
                m = re.search(r"([0-9]*\.?[0-9]+)", mpp_txt)
                if m:
                    mpp = float(m.group(1))
                    if "km" in mpp_txt.lower():
                        mpp *= 1000
            except Exception:
                mpp = None
        # Fallback by instrument
        if mpp is None:
            if inst == "OHRC":
                mpp = 0.32
            elif inst == "TMC-2":
                mpp = 5.0
            elif inst == "IIRS":
                mpp = 140.0
            elif inst == "DFSAR":
                mpp = 25.0
        return PDS4Metadata(product_id=pid, instrument=inst, bit_depth=16, width=w, height=h, bands=bands, resolution_mpp=mpp)

    def _read_raw_binary(self, img_path: Path, meta: PDS4Metadata) -> np.ndarray:
        # Handle 8/12/16-bit: 12-bit stored as 16-bit, scale later via _normalize
        # Detect actual dtype by file size vs expected
        import os
        fsize = os.path.getsize(str(img_path))
        # Try to infer bit depth from file size
        expected8 = meta.width * meta.height
        expected16 = expected8 * 2
        if fsize >= expected16 * 0.9:
            dtype = np.uint16
        elif fsize >= expected8 * 0.9:
            dtype = np.uint8
        else:
            dtype = np.uint16 if meta.bit_depth > 8 else np.uint8
        arr = np.fromfile(str(img_path), dtype=dtype)
        expected = meta.width * meta.height
        if arr.size >= expected:
            arr = arr[:expected].reshape(meta.height, meta.width)
        else:
            side = int(np.sqrt(arr.size))
            arr = arr[: side * side].reshape(side, side)
        # If 12-bit packed as 16-bit with max 4095, _normalize will do /4095 correctly
        return arr

    def _try_imread(self, path: Path) -> np.ndarray:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
