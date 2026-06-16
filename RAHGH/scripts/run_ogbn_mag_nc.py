"""
Standalone ogbn-mag node classification script.
Completely separate from the main train.py HGB pipeline.

Usage:
    python scripts/run_ogbn_mag_nc.py --seeds 3 --d 256 --K 2 --epochs 200
    python scripts/run_ogbn_mag_nc.py --seeds 10   # uses defaults
"""
import argparse, csv, os, sys, time, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Load ogbn_mag_loader without triggering src.data.__init__ (which needs torch_geometric) ──
import importlib
spec = importlib.util.spec_from_file_location(
    "ogbn_mag_loader",
    os.path.join(os.path.dirname(__file__), "..", "src", "data", "ogbn_mag_loader.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules["ogbn_mag_loader"] = mod
spec.loader.exec_module(mod)
load_ogbn_mag = mod.load_ogbn_mag

# ── RAHGH model imports (no torch_geometric dependency) ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model.rahgh import (
    compile_model, build_rahgh_classifier,
    build_edge_index_dict, build_node_type_indices,
)

# ── OGB evaluator ──
from ogb.nodeproppred import Evaluator
ogb_evaluator = Evaluator(name="ogbn-mag")


def evaluate(logits, target_size, idx, labels_full):
    p = logits[:target_size][idx].argmax(1).cpu().numpy()
    y = labels_full[idx].numpy()
    input_dict = {
        "y_true": torch.tensor(y).view(-1, 1),
        "y_pred": torch.tensor(p).view(-1, 1),
    }
    result = ogb_evaluator.eval(input_dict)
    return result["acc"]


def run_nc(data, params, seed=42, out_dir=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_dict = {k: v.to(device) for k, v in data["X_dict"].items()}
    edge_index_dict = build_edge_index_dict(data, device)
    node_type_indices = {k: v.to(device) for k, v in build_node_type_indices(data).items()}
    labels = data["labels"].to(device)
    Nt = data["target_size"]
    d = params["d"]

    model = build_rahgh_classifier(
        data, hidden_dim=d, num_classes=data["n_classes"],
        K=params["K"],
        dropout_homo=params["dropout"],
        dropout_gnn=params.get("dropout_gnn", params["dropout"]),
        gnn_hidden_dim=params.get("hidden", d),
    ).to(device)
    model = compile_model(model)
    opt = AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=params["epochs"], eta_min=params["lr"] * 0.01,
    )
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    # Use OGB splits
    train_idx = data["train_indices"].numpy()
    valid_idx = data["valid_indices"].numpy()
    test_idx = data["test_indices"].numpy()

    tr_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    va_t = torch.tensor(valid_idx, dtype=torch.long, device=device)
    te_t = torch.tensor(test_idx, dtype=torch.long, device=device)
    t0 = time.time()

    best_val_acc = 0.0
    best_sd = None
    stall = 0
    patience = 50
    epoch_rows = []
    pbar = tqdm(range(1, params["epochs"] + 1), desc="ogbn-mag NC")
    for ep in pbar:
        model.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits, *_ = model(x_dict, edge_index_dict, node_type_indices)
            loss = F.cross_entropy(logits[:Nt][tr_t], labels[tr_t])
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_ev, _ = model(x_dict, edge_index_dict, node_type_indices)
            preds = logits_ev[:Nt][tr_t].argmax(1).cpu().numpy()
            tr_acc = (preds == labels[tr_t].cpu().numpy()).mean()
            va_acc = evaluate(logits_ev, Nt, valid_idx, data["labels"])
        epoch_rows.append({"epoch": ep, "loss": loss.item(), "val_acc": va_acc})
        pbar.set_description(f"loss={loss.item():.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        logits, alpha = model(x_dict, edge_index_dict, node_type_indices)
        test_acc = evaluate(logits, Nt, test_idx, data["labels"])

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, f"final_model_seed{seed}.pt"))
        ep_path = Path(out_dir) / f"epoch_metrics_seed{seed}.csv"
        write_header = not ep_path.exists()
        with open(ep_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_acc"])
            if write_header:
                w.writeheader()
            w.writerows(epoch_rows)

    return dict(test_acc=test_acc, time_sec=time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dropout_gnn", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out", default="results/nc/ogbn_mag")
    args = parser.parse_args()

    print("=" * 60)
    print("  ogbn-mag — Node Classification (standalone)")
    print("=" * 60)

    data = load_ogbn_mag()
    data["name"] = "ogbn_mag"
    print(f"  N={data['N']}, target_size={data['target_size']}, n_classes={data['n_classes']}")
    print(f"  train={len(data['train_indices'])}, valid={len(data['valid_indices'])}, test={len(data['test_indices'])}")

    params = {
        "d": args.d, "K": args.K, "dropout": args.dropout,
        "dropout_gnn": args.dropout_gnn, "hidden": args.hidden,
        "lr": args.lr, "wd": args.wd, "epochs": args.epochs,
    }
    print(f"\n  Params: {params}")
    print(f"  Seeds: {args.seeds}\n")

    out_dir = args.out
    rows = []
    seed_list = [42 + 42 * i for i in range(args.seeds)]
    for seed in seed_list:
        r = run_nc(data, params, seed=seed, out_dir=out_dir)
        rows.append({
            "seed": seed,
            "test_acc": round(r["test_acc"], 4),
            "time_sec": round(r["time_sec"], 2),
        })
        print(f"  seed={seed:3d}  test_acc={r['test_acc']:.4f}  [{r['time_sec']:.0f}s]")

    accs = [r["test_acc"] for r in rows]
    print(f"\n  {'='*40}")
    print(f"  ogbn-mag NC  (n={args.seeds} seeds)")
    print(f"  Test Acc : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  {'='*40}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "test_acc", "time_sec"])
        w.writeheader()
        w.writerows(rows)
    summary = {
        "dataset": "ogbn_mag", "task": "nc",
        "acc_mean": round(float(np.mean(accs)), 4),
        "acc_sd": round(float(np.std(accs)), 4),
        "n_seeds": args.seeds,
    }
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary.keys())
        w.writeheader()
        w.writerow(summary)
    print(f"  Results → {out_dir}/")


if __name__ == "__main__":
    main()
