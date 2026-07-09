from .ogbn_mag_loader import load_ogbn_mag


def load_ogb(name: str, root: str = "data/raw") -> dict:
    name_lower = name.lower().replace("-", "_")
    LOADERS = {
        "ogbn_mag": lambda: load_ogbn_mag(root=root),
    }
    if name_lower in LOADERS:
        return LOADERS[name_lower]()
    raise ValueError(f"Unknown OGB dataset: {name}. Available: {list(LOADERS.keys())}")
