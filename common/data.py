"""Amazon Computers dataset loading and a fixed, shared train/valid/test split.

All models in this benchmark (graphsage/, sagat/, gamlp/) load the graph
through `load_products` and, by default, evaluate on the exact same
`split_idx.csv` split (see `--split-file` in each training script) so that
cross-model comparisons are fair.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch


def build_stratified_split(labels, train_ratio=0.6, valid_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    split_parts = {"train": [], "valid": [], "test": []}

    for cls in np.unique(labels_np):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)
        n_total = len(cls_idx)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)
        if n_train + n_valid >= n_total and n_total >= 3:
            n_train = max(1, n_total - 2)
            n_valid = 1
        elif n_train + n_valid >= n_total and n_total == 2:
            n_train = 1
            n_valid = 1
        elif n_total == 1:
            n_train = 1
            n_valid = 0
        split_parts["train"].append(cls_idx[:n_train])
        split_parts["valid"].append(cls_idx[n_train:n_train + n_valid])
        split_parts["test"].append(cls_idx[n_train + n_valid:])

    split_idx = {}
    for name, parts in split_parts.items():
        if any(len(part) > 0 for part in parts):
            idx = np.concatenate([part for part in parts if len(part) > 0])
        else:
            idx = np.array([], dtype=np.int64)
        rng.shuffle(idx)
        split_idx[name] = torch.from_numpy(idx).long()
    return split_idx


def _download_amazon_computer(root):
    """Download via PyG when the NPZ is missing (works on Kaggle and clean environments).
    torch-geometric is only imported here, lazily — it is an optional dependency
    used solely as a downloader, not for any model in this benchmark."""
    try:
        from torch_geometric.datasets import Amazon
        Amazon(root=str(root), name="Computers")
    except Exception as e:
        raise FileNotFoundError(
            f"amazon_co_buy_computer.npz not found under {root} and auto-download failed: {e}\n"
            "Place the file manually (see README) or `pip install torch-geometric` to auto-download."
        ) from e


def find_amazon_computer_npz(root):
    root = Path(root)
    candidates = sorted(root.glob("amazon_co_buy_computer*/amazon_co_buy_computer.npz"))
    candidates += sorted(root.glob("**/amazon_co_buy_computer.npz"))
    if not candidates:
        _download_amazon_computer(root)
        candidates = sorted(root.glob("amazon_co_buy_computer*/amazon_co_buy_computer.npz"))
        candidates += sorted(root.glob("**/amazon_co_buy_computer.npz"))
    if not candidates:
        raise FileNotFoundError(f"amazon_co_buy_computer.npz not found under {root}")
    return candidates[0]


def load_products(root, logger, split_seed=42):
    npz_path = find_amazon_computer_npz(root)
    with np.load(npz_path, allow_pickle=True) as loader:
        loader = dict(loader)

    adj = sp.csr_matrix(
        (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
        shape=tuple(loader["adj_shape"]),
    )
    adj = adj.maximum(adj.T).tocoo()
    edge_index = torch.from_numpy(np.vstack([adj.row, adj.col]).astype(np.int64, copy=False))

    if "attr_data" in loader:
        x = sp.csr_matrix(
            (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
            shape=tuple(loader["attr_shape"]),
        ).toarray()
    else:
        x = loader["attr_matrix"]

    labels = torch.as_tensor(loader["labels"], dtype=torch.long).view(-1)
    num_classes = int(len(np.unique(loader["labels"])))
    split_idx = build_stratified_split(labels, seed=split_seed)
    data = SimpleNamespace(
        x=torch.as_tensor(x, dtype=torch.float32),
        edge_index=edge_index.long(),
        num_nodes=int(labels.numel()),
        cache_name="amazon_computer",
    )
    logger.info(
        "Loaded Amazon Computers from %s | nodes=%d edges=%d train=%d valid=%d test=%d classes=%d",
        npz_path,
        data.num_nodes,
        data.edge_index.size(1),
        split_idx["train"].numel(),
        split_idx["valid"].numel(),
        split_idx["test"].numel(),
        num_classes,
    )
    return data, labels, split_idx, num_classes


class AccuracyEvaluator:
    def eval(self, input_dict):
        y_true = input_dict["y_true"].view(-1)
        y_pred = input_dict["y_pred"].view(-1)
        return {"acc": float((y_true == y_pred).float().mean().item())}


def load_split_idx_csv(path):
    """Load a pre-saved train/valid/test split from a two-column CSV (split,idx).

    The CSV must have a header row 'split,idx' followed by one row per node.
    Produced once by save_split_idx_csv() so every model evaluates on the
    exact same train/valid/test node sets — fair cross-model comparison.
    """
    parts = {"train": [], "valid": [], "test": []}
    with open(path, newline="", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            split, idx = line.split(",", 1)
            parts[split].append(int(idx))
    return {k: torch.tensor(v, dtype=torch.long) for k, v in parts.items()}


def save_split_idx_csv(split_idx, path):
    """Persist a split_idx dict to CSV so it can be reloaded with load_split_idx_csv."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("split,idx\n")
        for name in ("train", "valid", "test"):
            for idx in split_idx[name].tolist():
                f.write(f"{name},{idx}\n")
