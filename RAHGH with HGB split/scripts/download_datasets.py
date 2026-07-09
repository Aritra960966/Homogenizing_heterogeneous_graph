import os
import sys
import json
import shutil
import zipfile
import tarfile
import hashlib
import logging
import argparse
import requests
from pathlib import Path
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _download_file(url: str, dest: Path, desc: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info(f"  already exists: {dest.name}  (skipping)")
        return dest

    log.info(f"  downloading {desc or dest.name} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=desc or dest.name, leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def _extract(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  extracting {archive.name} -> {dest_dir} ...")

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dest_dir)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(dest_dir)
    elif archive.name.endswith(".tar.bz2"):
        with tarfile.open(archive, "r:bz2") as t:
            t.extractall(dest_dir)
    else:
        log.warning(f"  unknown archive format: {archive.name}")


def download_dblp() -> bool:
    log.info("=== DBLP ===")
    dest = RAW / "dblp"
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        log.info("  DBLP already downloaded.")
        return True

    try:
        from torch_geometric.datasets import DBLP as PyGDBLP
        ds = PyGDBLP(root=str(dest))
        log.info(f"  DBLP loaded: {ds[0]}")
        marker.touch()
        return True
    except Exception as e:
        log.error(f"  PyG DBLP download failed: {e}")
        log.info("  Trying HGB fallback ...")
        return _download_hgb_dataset("DBLP", dest)


def download_acm() -> bool:
    log.info("=== ACM ===")
    dest = RAW / "acm"
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        log.info("  ACM already downloaded.")
        return True

    try:
        from torch_geometric.datasets import ACM as PyGACM
        ds = PyGACM(root=str(dest))
        log.info(f"  ACM loaded: {ds[0]}")
        marker.touch()
        return True
    except Exception as e:
        log.error(f"  PyG ACM download failed: {e}")
        log.info("  Trying HGB fallback ...")
        return _download_hgb_dataset("ACM", dest)


def download_imdb() -> bool:
    log.info("=== IMDB ===")
    dest = RAW / "imdb"
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        log.info("  IMDB already downloaded.")
        return True

    try:
        from torch_geometric.datasets import IMDB as PyGIMDB
        ds = PyGIMDB(root=str(dest))
        log.info(f"  IMDB loaded: {ds[0]}")
        marker.touch()
        return True
    except Exception as e:
        log.error(f"  PyG IMDB download failed: {e}")
        log.info("  Trying HGB fallback ...")
        return _download_hgb_dataset("IMDB", dest)


def download_ogbn_mag() -> bool:
    log.info("=== OGBN-MAG ===")
    dest = RAW / "ogbn_mag"
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        log.info("  OGBN-MAG already downloaded.")
        return True

    try:
        from ogb.nodeproppred import PygNodePropPredDataset
        ds = PygNodePropPredDataset(name="ogbn-mag", root=str(dest))
        log.info(f"  OGBN-MAG loaded: {ds[0]}")
        marker.touch()
        return True
    except ImportError:
        log.error("  OGB not installed. Run: pip install ogb")
        return False
    except Exception as e:
        log.error(f"  OGBN-MAG download failed: {e}")
        return False


def download_amazon() -> bool:
    log.info("=== Amazon ===")
    dest = RAW / "amazon"
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        log.info("  Amazon already downloaded.")
        return True

    try:
        from torch_geometric.datasets import Amazon
        ds = Amazon(root=str(dest), name="Computers")
        log.info(f"  Amazon Computers loaded: {ds[0]}")
        marker.touch()
        return True
    except Exception as e:
        log.warning(f"  PyG Amazon failed: {e}")

    return _download_hgb_dataset("Amazon", dest)


def download_lastfm() -> bool:
    log.info("=== LastFM ===")
    dest = RAW / "lastfm"
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        log.info("  LastFM already downloaded.")
        return True

    return _download_hgb_dataset("LastFM", dest)


def download_yelp() -> bool:
    log.info("=== Yelp ===")
    dest = RAW / "yelp"
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        log.info("  Yelp already downloaded.")
        return True

    return _download_hgb_dataset("Yelp", dest)


def download_freebase() -> bool:
    log.info("=== Freebase ===")
    dest = RAW / "freebase"
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        log.info("  Freebase already downloaded.")
        return True

    return _download_hgb_dataset("Freebase", dest)


HGB_BASE = "https://cloud.tsinghua.edu.cn/d/2d965df24d6e4f129710/files/?p=%2F"
HGB_URLS = {
    "DBLP":     "https://cloud.tsinghua.edu.cn/f/a0ac2498f3c842419da6/?dl=1",
    "ACM":      "https://cloud.tsinghua.edu.cn/f/f3b6b3e6c78843e0aabd/?dl=1",
    "IMDB":     "https://cloud.tsinghua.edu.cn/f/fbdc9f8c38474e79ad8e/?dl=1",
    "Amazon":   "https://cloud.tsinghua.edu.cn/f/4d1b6ee1ddc64f279d17/?dl=1",
    "LastFM":   "https://cloud.tsinghua.edu.cn/f/ba49e4e56ecf4c0fa5e2/?dl=1",
    "Yelp":     "https://cloud.tsinghua.edu.cn/f/2e2a69aa7cc847279073/?dl=1",
    "Freebase": "https://cloud.tsinghua.edu.cn/f/d8e7b82264c8442d88f3/?dl=1",
}


def _download_hgb_dataset(name: str, dest: Path) -> bool:
    if name not in HGB_URLS:
        log.error(f"  No HGB URL registered for {name}")
        return False

    url = HGB_URLS[name]
    archive = dest / f"{name.lower()}.zip"

    try:
        _download_file(url, archive, desc=f"HGB/{name}")
        _extract(archive, dest)
        archive.unlink()
        (dest / ".done").touch()
        log.info(f"  {name} downloaded via HGB ✓")
        return True
    except Exception as e:
        log.error(f"  HGB download failed for {name}: {e}")
        return False


def write_manifest(results: dict) -> None:
    manifest_path = RAW / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nManifest written -> {manifest_path}")


DATASETS = {
    "dblp":     download_dblp,
    "acm":      download_acm,
    "imdb":     download_imdb,
    "ogbn_mag": download_ogbn_mag,
    "amazon":   download_amazon,
    "lastfm":   download_lastfm,
    "yelp":     download_yelp,
    "freebase": download_freebase,
}


def main():
    parser = argparse.ArgumentParser(description="Download RAHGH datasets")
    parser.add_argument("--only",  nargs="+", choices=DATASETS.keys(),
                        help="Download only these datasets")
    parser.add_argument("--skip",  nargs="+", choices=DATASETS.keys(),
                        help="Skip these datasets")
    args = parser.parse_args()

    to_run = list(DATASETS.keys())
    if args.only:
        to_run = [d for d in args.only if d in DATASETS]
    if args.skip:
        to_run = [d for d in to_run if d not in args.skip]

    log.info(f"Datasets to download: {to_run}")
    log.info(f"Raw data root: {RAW}\n")

    results = {}
    for name in to_run:
        try:
            ok = DATASETS[name]()
            results[name] = "ok" if ok else "failed"
        except Exception as e:
            log.error(f"Unexpected error for {name}: {e}")
            results[name] = f"error: {e}"

    write_manifest(results)

    print("\n" + "=" * 50)
    print("Download summary")
    print("=" * 50)
    for name, status in results.items():
        icon = "✓" if status == "ok" else "✗"
        print(f"  {icon}  {name:<15} {status}")
    print("=" * 50)

    failed = [k for k, v in results.items() if v != "ok"]
    if failed:
        print(f"\nFailed: {failed}")
        print("Check logs above. Common fixes:")
        print("  pip install torch torch-geometric ogb requests tqdm")
        sys.exit(1)
    else:
        print("\nAll datasets downloaded successfully.")


if __name__ == "__main__":
    main()
