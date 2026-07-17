"""Shared utilities for every training script in this benchmark
(graphsage/, sagat/, gamlp/).

Every model here is built from plain PyTorch tensor ops (scatter_add_ /
scatter_reduce) instead of compiled message-passing kernels (torch-scatter /
torch-sparse / torch-geometric.nn / DGL). Those compiled extensions must
match the installed torch build exactly, which breaks easily across
environments (CPU-only boxes, Kaggle, mismatched CUDA versions). Avoiding
them keeps every script here runnable with nothing beyond `torch` itself,
and keeps peak memory at O(E) rather than O(N^2) for the full Amazon
Computers graph (~13.7K nodes).
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu):
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def make_output_dir(base_dir, run_name):
    out_dir = Path(base_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def setup_logger(name, out_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "train.log")]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def plot_training_curves(metrics_path, out_dir, title=None):
    """Plot loss and accuracy per epoch from metrics.jsonl, save curves.png to out_dir.

    Each jsonl line needs keys: epoch, loss, train_acc, val_acc (may be None),
    best_test. The "stage" key (multi-stage GAMLP training) is optional — stage
    boundaries are marked with a vertical line, x-axis runs continuously across
    stages. Returns the image path, or None if matplotlib is missing / no data.
    """
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return None
    records = [json.loads(line) for line in
               metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.getLogger(__name__).warning(
            "matplotlib not installed — skipping training curve plot")
        return None

    xs        = list(range(len(records)))
    loss      = [r.get("loss") for r in records]
    train_acc = [r.get("train_acc") for r in records]
    val_pts   = [(i, r["val_acc"]) for i, r in enumerate(records)
                 if r.get("val_acc") is not None]
    best_test = records[-1].get("best_test")

    stage_starts = [i for i in range(1, len(records))
                    if records[i].get("stage") != records[i - 1].get("stage")]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(xs, loss, color="tab:red", linewidth=1.2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss")
    ax_loss.set_title("Loss")

    ax_acc.plot(xs, train_acc, color="tab:blue", linewidth=1.2, label="train")
    if val_pts:
        vx, vy = zip(*val_pts)
        ax_acc.plot(vx, vy, color="tab:orange", linewidth=1.2, label="valid")
        bi = max(range(len(vy)), key=lambda k: vy[k])
        ax_acc.scatter([vx[bi]], [vy[bi]], color="tab:orange", zorder=3,
                       label=f"best valid = {vy[bi]:.4f}")
    if best_test is not None and best_test >= 0:
        ax_acc.axhline(best_test, color="tab:green", linestyle="--", linewidth=1,
                       label=f"best test = {best_test:.4f}")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend(loc="lower right", fontsize=8)

    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        for s in stage_starts:
            ax.axvline(s, color="gray", linestyle=":", linewidth=1)
    if stage_starts:
        ax_loss.set_xlabel("Epoch (continuous across stages)")
        ax_acc.set_xlabel("Epoch (continuous across stages)")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_dir) / "curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# Pure-PyTorch graph ops (no torch-scatter / torch-sparse / torch-geometric dependency)
# ----------------------------------------------------------------------------

def scatter_add(src, index, dim_size):
    """Sum rows of `src` into `dim_size` buckets given by `index` (grouping on dim 0)."""
    shape = (dim_size,) + src.shape[1:]
    out = src.new_zeros(shape)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_mean(src, index, dim_size):
    summed = scatter_add(src, index, dim_size)
    count = scatter_add(src.new_ones(src.size(0), 1), index, dim_size).clamp(min=1)
    return summed / count


def scatter_softmax(src, index, dim_size):
    """Softmax of 1-D `src` within each group defined by `index` (used by GAT / SA-GAT
    attention, where the "group" is the set of edges pointing at one target node)."""
    src_max = src.new_full((dim_size,), float("-inf"))
    src_max = src_max.scatter_reduce(0, index, src, reduce="amax", include_self=True)
    src_max = torch.nan_to_num(src_max, neginf=0.0)
    shifted = (src - src_max[index]).exp()
    denom = scatter_add(shifted.unsqueeze(-1), index, dim_size).squeeze(-1).clamp(min=1e-16)
    return shifted / denom[index]


def add_self_loops(edge_index, num_nodes):
    loop = torch.arange(num_nodes, device=edge_index.device)
    return torch.cat([edge_index, torch.stack([loop, loop], dim=0)], dim=1)


def gcn_norm_edge_weight(edge_index, num_nodes):
    """Symmetric D^-1/2 A D^-1/2 edge weights (Kipf & Welling). `edge_index` must
    already include self loops."""
    row, col = edge_index
    deg = scatter_add(row.new_ones(row.size(0), 1, dtype=torch.float32), row, num_nodes).squeeze(-1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    return deg_inv_sqrt[row] * deg_inv_sqrt[col]


def scatter_max_feat(src, index, dim_size):
    """Max-aggregation over feature vectors using scatter_reduce_ (PyTorch 1.12+).
    Nodes with no incoming edges keep the initial value of 0."""
    out = src.new_zeros(dim_size, src.size(-1))
    out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src, reduce="amax", include_self=False)
    return out


def scatter_std_feat(src, index, dim_size, mean=None):
    """Std-dev aggregation: sqrt(E[x^2] - E[x]^2). Uses a pre-computed mean if supplied."""
    if mean is None:
        mean = scatter_mean(src, index, dim_size)
    sq_mean = scatter_mean(src * src, index, dim_size)
    # eps keeps variance > 0: sqrt's gradient at 0 is infinite -> NaN loss otherwise
    return ((sq_mean - mean * mean).clamp(min=0) + 1e-6).sqrt()


def build_adjacency_list(edge_index, num_nodes):
    """Python list of 1-D LongTensors: neighbours of each node. Used by mini-batch
    neighbour-sampling modes (graphsage/train_graphsage.py, train_improved_graphsage.py)."""
    row, col = edge_index.cpu()
    order = torch.argsort(row)
    row, col = row[order], col[order]
    counts = torch.bincount(row, minlength=num_nodes).tolist()
    adj, ptr = [], 0
    for c in counts:
        adj.append(col[ptr:ptr + c])
        ptr += c
    return adj


def propagate_mean(edge_index, num_nodes, feat, deg=None, hops=1):
    """Repeated 1-hop mean propagation: feat <- scatter_mean(feat[neighbours]) `hops` times.

    Used by the GAMLP family (gamlp/) to precompute K-hop smoothed features and
    label embeddings. `edge_index` must be symmetric (undirected) — this graph's
    loader always symmetrises adjacency, so aggregating "row"-into or "col"-into
    buckets gives the same result and a plain scatter_add replaces what the
    original GAMLP code did with a torch-sparse SparseTensor matmul.
    """
    row, col = edge_index
    if deg is None:
        deg = scatter_add(row.new_ones(row.size(0), 1, dtype=torch.float32), row, num_nodes).clamp(min=1)
    for _ in range(hops):
        feat = scatter_add(feat[col], row, num_nodes) / deg
    return feat
