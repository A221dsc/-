"""Fetch and verify original dataset archives without extracting member paths."""
import argparse
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from prepare import PORTO, TDRIVE, json_file, sha

SOURCES = {
    "tdrive": (TDRIVE, "https://1drv.ms/u/s%21AsWQUIUFkRXPgsN6OCa9OB-qziGfhA?download=1",
               "MSR-LA; academic noncommercial use, no redistribution of data or derivatives",
               "2011-08 T-Drive sample"),
    "porto": (PORTO, "https://archive.ics.uci.edu/static/public/339/taxi%2Bservice%2Btrajectory%2Bprediction%2Bchallenge%2Becml%2Bpkdd%2B2015.zip",
              "CC BY 4.0", "UCI dataset 339"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=SOURCES)
    p.add_argument("--output", type=Path, default=Path("data/raw"))
    p.add_argument("--local-archive", type=Path, help="verify a separately downloaded official ZIP")
    p.add_argument("--sha256", help="expected publisher or independently recorded SHA-256, if available")
    args = p.parse_args()
    page, url, license_name, version = SOURCES[args.dataset]
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / f"{args.dataset}.zip"
    source = args.local_archive
    if source is None:
        if dest.exists():
            source = dest
        else:
            source = dest.with_suffix(".download")
            subprocess.run(["curl.exe" if __import__("os").name == "nt" else "curl",
                            "-fL", "--retry", "2", "--max-time", "1200",
                            "-o", str(source), url], check=True)
    digest = sha(source)
    if args.sha256 and args.sha256.lower() != digest:
        raise ValueError("archive SHA-256 mismatch")
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        expected = "train.csv.zip" in names if args.dataset == "porto" else any(
            Path(n).suffix.lower() == ".txt" and Path(n).stem.isdecimal() for n in names)
        if not expected or z.testzip() is not None:
            raise ValueError("archive member layout or CRC failed")
    if args.local_archive is None and source != dest:
        source.replace(dest)
        source = dest
    json_file(args.output / f"{args.dataset}_source.json", {
        "official_page": page, "download_url": url, "version": version, "license": license_name,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "bytes": source.stat().st_size, "sha256": digest, "file": source.name,
        "zip_crc_ok": True, "compared_with_expected_sha256": bool(args.sha256),
        "local_archive_supplied": args.local_archive is not None,
    })
    print(f"{source}: {source.stat().st_size} bytes, SHA256 {digest}, ZIP CRC OK")


if __name__ == "__main__":
    main()
