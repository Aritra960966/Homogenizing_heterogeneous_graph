import os
import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CHECKS = {
    "dblp": {
        "expected_files": ["DBLP/processed", "DBLP/raw"],
        "pyg_loader": ("torch_geometric.datasets", "DBLP"),
        "node_types": ["author", "paper", "term", "conference"],
        "min_nodes": 10000,
    },
    "acm": {
        "expected_files": ["ACM/processed", "ACM/raw"],
        "pyg_loader": ("torch_geometric.datasets", "ACM"),
        "node_types": ["paper", "author", "subject"],
        "min_nodes": 5000,
    },
    "imdb": {
        "expected_files": ["IMDB/processed", "IMDB/raw"],
        "pyg_loader": ("torch_geometric.datasets", "IMDB"),
        "node_types": ["movie", "director", "actor"],
        "min_nodes": 3000,
    },
    "ogbn_mag": {
        "expected_files": ["ogbn_mag"],
        "pyg_loader": None,
        "ogb_loader": ("ogb.nodeproppred", "PygNodePropPredDataset", "ogbn-mag"),
        "node_types": ["paper", "author", "institution", "field_of_study"],
        "min_nodes": 1000000,
    },
    "amazon": {
        "expected_files": ["Amazon"],
        "pyg_loader": None,
        "min_nodes": 5000,
    },
    "lastfm": {
        "expected_files": [],
        "pyg_loader": None,
        "min_nodes": 0,
    },
}


def check_marker(name: str) -> bool:
    marker = RAW / name / ".done"
    if marker.exists():
        log.info(f"  \u2713 .done marker present")
        return True
    log.warning(f"  \u2717 .done marker missing")
    return False


def check_dir_size(name: str) -> None:
    d = RAW / name
    if not d.exists():
        log.warning(f"  \u2717 directory not found: {d}")
        return
    files = list(d.rglob("*"))
    total = sum(f.stat().st_size for f in files if f.is_file())
    log.info(f"  directory: {d}")
    log.info(f"  files:     {len(files)}")
    log.info(f"  size:      {total/1e6:.1f} MB")


def load_and_report_pyg(name: str, module: str, cls: str) -> None:
    try:
        import importlib
        mod = importlib.import_module(module)
        DatasetCls = getattr(mod, cls)
        ds = DatasetCls(root=str(RAW / name))
        data = ds[0]
        log.info(f"  PyG load OK: {data}")
        if hasattr(data, "node_types"):
            log.info(f"  node types: {data.node_types}")
        if hasattr(data, "edge_types"):
            log.info(f"  edge types: {len(data.edge_types)} relations")
    except Exception as e:
        log.warning(f"  PyG load failed: {e}")


def load_and_report_ogb(name: str, module: str, cls: str,
                        ogb_name: str) -> None:
    try:
        import importlib
        mod = importlib.import_module(module)
        DatasetCls = getattr(mod, cls)
        ds = DatasetCls(name=ogb_name, root=str(RAW / name))
        data = ds[0]
        log.info(f"  OGB load OK: {data}")
        split_idx = ds.get_idx_split()
        log.info(f"  train/val/test: {len(split_idx['train']['paper'])} / "
                 f"{len(split_idx['valid']['paper'])} / "
                 f"{len(split_idx['test']['paper'])}")
    except Exception as e:
        log.warning(f"  OGB load failed: {e}")


def verify(name: str) -> None:
    log.info(f"\n{'=' * 50}")
    log.info(f"  Dataset: {name.upper()}")
    log.info(f"{'=' * 50}")

    check_marker(name)
    check_dir_size(name)

    cfg = CHECKS.get(name, {})

    if cfg.get("pyg_loader"):
        module, cls = cfg["pyg_loader"]
        load_and_report_pyg(name, module, cls)

    if cfg.get("ogb_loader"):
        module, cls, ogb_name = cfg["ogb_loader"]
        load_and_report_ogb(name, module, cls, ogb_name)


def main():
    parser = argparse.ArgumentParser(description="Verify raw dataset downloads")
    parser.add_argument("--dataset", nargs="+",
                        choices=["dblp","acm","imdb","ogbn_mag",
                                 "amazon","lastfm","yelp","freebase"],
                        help="Datasets to verify (default: all)")
    args = parser.parse_args()

    datasets = args.dataset or ["dblp","acm","imdb","ogbn_mag",
                                "amazon","lastfm","yelp","freebase"]

    for name in datasets:
        verify(name)

    log.info(f"\n{'=' * 50}")
    log.info("Verification complete.")


if __name__ == "__main__":
    main()
