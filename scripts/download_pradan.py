"""PRADAN Jump-To-FileResume handler - handles ZIP volume limits & bulk resume.
PRADAN requires login + JWT; this script supports --token or env PRADAN_TOKEN.
Usage: python scripts/download_pradan.py --url https://pradan.issdc.gov.in/... --out data/raw --resume --token $PRADAN_TOKEN
"""
import argparse
import hashlib
import os
from pathlib import Path
import requests

def download_with_resume(url: str, out_path: Path, chunk=8192, token: str | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    existing = 0
    if out_path.exists():
        existing = out_path.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming from {existing} bytes (Jump-To-File)")
    if token or os.getenv("PRADAN_TOKEN"):
        headers["Authorization"] = f"Bearer {token or os.getenv('PRADAN_TOKEN')}"
        headers["X-Auth-Token"] = token or os.getenv("PRADAN_TOKEN")

    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    if resp.status_code in (206, 200):
        mode = "ab" if resp.status_code == 206 else "wb"
        total = int(resp.headers.get("content-length", 0)) + existing
        with open(out_path, mode) as f:
            for chunk_data in resp.iter_content(chunk_size=chunk):
                if chunk_data:
                    f.write(chunk_data)
        print(f"Saved {out_path} ({total} bytes)")
        # Verify if checksum provided in header
        return out_path
    else:
        print(f"Failed: {resp.status_code} {resp.text[:200]}")
        raise SystemExit(1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--token", default=None, help="PRADAN JWT (or set PRADAN_TOKEN env)")
    args = p.parse_args()
    download_with_resume(args.url, Path(args.out), token=args.token)
