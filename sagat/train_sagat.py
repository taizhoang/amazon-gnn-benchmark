"""Structure-Aware GAT (SA-GAT) on Amazon Computers.

Motivation: plain GAT computes attention purely from node content
(e^T [W h_i || W h_j]), so two neighbours with similar features but very
different roles in the graph (e.g. a hub product vs. a niche one) get
attention scores that don't reflect that difference at all. SA-GAT adds four
structural statistics per node — degree, PageRank, clustering coefficient,
and (sampled) betweenness centrality — as an extra additive term in the
attention logit, so the model can learn to weight neighbours differently
based on their structural role, not just their content.

This is a from-scratch pure-PyTorch reimplementation of the DGL-based
Structure-Aware GAT prototype (GRAPH-SAGAT/run_structure_aware_gat.py),
rebuilt on this benchmark's shared scatter_add / scatter_softmax attention
machinery (see graphsage-sibling train_gat-style layer below) so it needs no
DGL / torch-scatter / torch-sparse dependency, matching every other script
in this repo.

Three comparable variants, selected with --variant:
  gat         baseline GAT — no structural information at all.
  gat-concat  structural features concatenated onto the raw node input
              (ablation: does giving the model structural info at all help,
              regardless of *how* it's used?).
  sagat       structural features injected as an additive bias inside the
              attention logit itself (ours) — same information as
              gat-concat, but used to *reweight neighbours* rather than
              *describe the node*.

When --variant sagat, the run additionally measures how strongly the
layer-1 attention weights correlate (Pearson r) with each neighbour's raw
structural statistic — the analysis GRAPH-SAGAT used to check whether the
model actually learned to lean on structure.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.data import AccuracyEvaluator, load_products, load_split_idx_csv
from common.utils import (
    add_self_loops, append_jsonl, count_params, get_device, make_output_dir,
    plot_training_curves, scatter_add, scatter_softmax, set_seed, setup_logger,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Structure-Aware GAT (SA-GAT) on Amazon Computers")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/sagat")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--variant", choices=["gat", "gat-concat", "sagat"], default="sagat",
                        help="gat: vanilla baseline (no structural info). "
                             "gat-concat: structural features concatenated onto node input (ablation). "
                             "sagat: structural features injected as an additive attention bias (ours).")

    parser.add_argument("--head-dim", type=int, default=32, help="output dim per attention head")
    parser.add_argument("--heads", type=int, default=8, help="attention heads in the hidden layer(s)")
    parser.add_argument("--out-heads", type=int, default=2, help="attention heads in the output layer (averaged)")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--struct-embed-dim", type=int, default=16, help="hidden width of the structural-feature MLP (sagat only)")
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--att-dropout", type=float, default=0.6, help="dropout on attention coefficients")
    parser.add_argument("--negative-slope", type=float, default=0.2)
    parser.add_argument("--betweenness-samples", type=int, default=500,
                        help="Nodes sampled for approximate betweenness centrality (exact is O(N*E))")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--split-file", type=str, default="split_idx.csv",
                        help="Pre-saved CSV split for fair cross-model comparison. "
                             "Empty string to regenerate from --seed.")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Structural features (degree, PageRank, clustering coefficient, betweenness)
# ----------------------------------------------------------------------------

def log_zscore(x):
    x = np.log1p(x)
    return (x - x.mean()) / (x.std() + 1e-8)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def compute_structural_features(edge_index, num_nodes, betweenness_samples, seed, logger):
    """Returns (normalised_feats [N,4] tensor, raw_stats dict of np arrays).
    Heavy-tailed stats (degree, betweenness) get a log-zscore; bounded ones
    (PageRank, clustering) get a plain zscore — same convention as the
    original GRAPH-SAGAT prototype."""
    row, col = edge_index.cpu().numpy()
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(zip(row.tolist(), col.tolist()))
    g.remove_edges_from(nx.selfloop_edges(g))

    t0 = time.time()
    degree_dict = dict(g.degree())
    degree = np.array([degree_dict[i] for i in range(num_nodes)], dtype=np.float32)
    pagerank_dict = nx.pagerank(g, alpha=0.85, max_iter=100, tol=1e-6)
    pagerank = np.array([pagerank_dict[i] for i in range(num_nodes)], dtype=np.float32)
    clustering_dict = nx.clustering(g)
    clustering = np.array([clustering_dict[i] for i in range(num_nodes)], dtype=np.float32)
    betweenness_dict = nx.betweenness_centrality(
        g, k=min(betweenness_samples, num_nodes), seed=seed, normalized=True
    )
    betweenness = np.array([betweenness_dict[i] for i in range(num_nodes)], dtype=np.float32)
    logger.info("Structural features computed in %.1fs", time.time() - t0)

    feats = np.stack(
        [log_zscore(degree), zscore(pagerank), zscore(clustering), log_zscore(betweenness)],
        axis=1,
    )
    raw = {"degree": degree, "pagerank": pagerank, "clustering": clustering, "betweenness": betweenness}
    return torch.from_numpy(feats).float(), raw


# ----------------------------------------------------------------------------
# Model — same edge-softmax attention machinery as a plain GAT (scatter_softmax
# over incoming edges per target node), extended with an optional additive
# structural-attention bias term (SA-GAT) computed from `struct_embed`.
# ----------------------------------------------------------------------------

class GATLayer(nn.Module):
    def __init__(self, in_dim, head_dim, heads, dropout, att_dropout, negative_slope,
                concat=True, struct_dim=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.concat = concat
        self.struct_dim = struct_dim
        self.lin = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(heads * head_dim if concat else head_dim))
        if struct_dim is not None:
            self.att_src_s = nn.Parameter(torch.empty(heads, struct_dim))
            self.att_dst_s = nn.Parameter(torch.empty(heads, struct_dim))
            nn.init.xavier_uniform_(self.att_src_s)
            nn.init.xavier_uniform_(self.att_dst_s)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, num_nodes, struct_embed=None, return_attn=False):
        row, col = edge_index  # row = target (aggregator), col = source (neighbour)
        x = self.dropout(x)
        h = self.lin(x).view(-1, self.heads, self.head_dim)  # [N, heads, head_dim]

        alpha_src = (h * self.att_src).sum(dim=-1)  # [N, heads]
        alpha_dst = (h * self.att_dst).sum(dim=-1)  # [N, heads]
        if self.struct_dim is not None:
            assert struct_embed is not None, "struct_embed required when struct_dim is set"
            alpha_src = alpha_src + (struct_embed.unsqueeze(1) * self.att_src_s).sum(dim=-1)
            alpha_dst = alpha_dst + (struct_embed.unsqueeze(1) * self.att_dst_s).sum(dim=-1)
        edge_alpha = F.leaky_relu(alpha_src[col] + alpha_dst[row], self.negative_slope)  # [E, heads]

        edge_alpha = torch.stack([
            scatter_softmax(edge_alpha[:, k], row, num_nodes) for k in range(self.heads)
        ], dim=-1)  # [E, heads]
        edge_alpha_dropped = self.att_dropout(edge_alpha)

        msg = h[col] * edge_alpha_dropped.unsqueeze(-1)  # [E, heads, head_dim]
        out = scatter_add(msg.reshape(msg.size(0), -1), row, num_nodes).view(num_nodes, self.heads, self.head_dim)

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.head_dim)
        else:
            out = out.mean(dim=1)
        out = out + self.bias
        if return_attn:
            return out, edge_alpha  # un-dropped attention, for analysis
        return out


class GAT(nn.Module):
    def __init__(self, in_dim, head_dim, out_dim, num_layers, heads, out_heads,
                dropout, att_dropout, negative_slope, struct_dim=None, struct_embed_dim=16):
        super().__init__()
        self.use_struct_attn = struct_dim is not None
        if self.use_struct_attn:
            self.struct_mlp = nn.Sequential(
                nn.Linear(struct_dim, struct_embed_dim), nn.ReLU(), nn.Dropout(dropout),
            )
            sdim = struct_embed_dim
        else:
            self.struct_mlp = None
            sdim = None

        self.layers = nn.ModuleList()
        dim = in_dim
        for _ in range(num_layers - 1):
            self.layers.append(GATLayer(dim, head_dim, heads, dropout, att_dropout, negative_slope,
                                        concat=True, struct_dim=sdim))
            dim = head_dim * heads
        self.layers.append(GATLayer(dim, out_dim, out_heads, dropout, att_dropout, negative_slope,
                                    concat=False, struct_dim=sdim))
        self.elu = nn.ELU()

    def forward(self, x, edge_index, num_nodes, struct_feats=None, return_attn=False):
        se = self.struct_mlp(struct_feats) if self.use_struct_attn else None
        attn_first = None
        for i, layer in enumerate(self.layers[:-1]):
            if return_attn and i == 0:
                x, attn_first = layer(x, edge_index, num_nodes, struct_embed=se, return_attn=True)
            else:
                x = layer(x, edge_index, num_nodes, struct_embed=se)
            x = self.elu(x)
        out = self.layers[-1](x, edge_index, num_nodes, struct_embed=se)
        if return_attn:
            return out, attn_first
        return out


def train_epoch(model, x, edge_index, num_nodes, labels, train_idx, optimizer, evaluator, struct_feats):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, num_nodes, struct_feats=struct_feats)
    loss = F.cross_entropy(out[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    pred = out[train_idx].argmax(dim=-1).detach().cpu()
    acc = evaluator.eval({"y_true": labels[train_idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]
    return float(loss.item()), acc


@torch.no_grad()
def evaluate(model, x, edge_index, num_nodes, labels, idx, evaluator, struct_feats):
    model.eval()
    out = model(x, edge_index, num_nodes, struct_feats=struct_feats)
    pred = out[idx].argmax(dim=-1).cpu()
    return evaluator.eval({"y_true": labels[idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]


@torch.no_grad()
def analyze_structural_attention(model, x, edge_index, num_nodes, struct_feats, raw_struct, out_dir, logger):
    """Pearson correlation between layer-1 attention weight and each raw
    structural statistic of the *source* (neighbour) node — does the model
    actually lean on structure, and which statistic does it lean on most?"""
    model.eval()
    _, attn = model(x, edge_index, num_nodes, struct_feats=struct_feats, return_attn=True)
    if attn is None:
        return None

    from scipy.stats import pearsonr

    alpha = attn.mean(dim=1).cpu().numpy()  # [E], averaged over heads
    row, col = edge_index.cpu().numpy()     # row = target, col = source
    mask = row != col                       # drop self-loops
    alpha = alpha[mask]
    src = col[mask]

    correlations = {}
    for name in ("degree", "pagerank", "clustering", "betweenness"):
        r, p = pearsonr(raw_struct[name][src], alpha)
        correlations[name] = {"r": float(r), "p": float(p)}
        logger.info("attention vs %-11s : r=%+.4f (p=%.2e)", name, r, p)
    write_json(out_dir / "attention_structure_correlation.json", correlations)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame({
            "alpha": alpha,
            "degree": raw_struct["degree"][src],
            "clustering": raw_struct["clustering"][src],
        })
        df["degree_q"] = pd.qcut(df["degree"].rank(method="first"), 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
        df["clustering_q"] = pd.qcut(df["clustering"].rank(method="first"), 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        df.groupby("degree_q", observed=True)["alpha"].mean().plot(kind="bar", ax=axes[0], color="#3B6D11")
        axes[0].set_title("Mean attention by neighbour degree quartile")
        axes[0].set_ylabel("Mean attention weight")
        axes[0].tick_params(axis="x", rotation=20)
        df.groupby("clustering_q", observed=True)["alpha"].mean().plot(kind="bar", ax=axes[1], color="#3B6D11")
        axes[1].set_title("Mean attention by neighbour clustering quartile")
        axes[1].tick_params(axis="x", rotation=20)
        plt.tight_layout()
        fig.savefig(out_dir / "attention_structure.png", dpi=150)
        plt.close(fig)
    except ImportError:
        logger.warning("matplotlib/pandas not installed — skipping attention-structure plot")

    return correlations


def run_once(args, run_id, device):
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}_{args.variant}"
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger("sagat", out_dir)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))
    set_seed(args.seed + run_id)

    data, labels, split_idx, num_classes = load_products(args.dataset_root, logger, split_seed=args.seed)
    split_file = Path(args.split_file) if args.split_file else None
    if split_file and split_file.is_file():
        split_idx = load_split_idx_csv(split_file)
        logger.info("Fixed split from %s  train=%d valid=%d test=%d",
                    split_file, split_idx["train"].numel(), split_idx["valid"].numel(), split_idx["test"].numel())

    struct_feats_norm = raw_struct = None
    if args.variant in ("gat-concat", "sagat"):
        struct_feats_norm, raw_struct = compute_structural_features(
            data.edge_index, data.num_nodes, args.betweenness_samples, args.seed, logger,
        )

    x = data.x.float()
    if args.variant == "gat-concat":
        x = torch.cat([x, struct_feats_norm], dim=-1)
    x = x.to(device)

    edge_index = add_self_loops(data.edge_index, data.num_nodes).to(device)
    labels = labels.to(device)
    split_idx = {k: v.to(device) for k, v in split_idx.items()}
    evaluator = AccuracyEvaluator()
    struct_dev = struct_feats_norm.to(device) if args.variant == "sagat" else None

    model = GAT(
        x.size(1), args.head_dim, num_classes, args.num_layers, args.heads, args.out_heads,
        args.dropout, args.att_dropout, args.negative_slope,
        struct_dim=(struct_feats_norm.size(1) if args.variant == "sagat" else None),
        struct_embed_dim=args.struct_embed_dim,
    ).to(device)
    logger.info("Model params: %d", count_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = best_test = -1.0
    best_epoch = stale = 0
    ckpt_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.jsonl"

    for epoch in range(args.epochs):
        t0 = time.time()
        loss, train_acc = train_epoch(model, x, edge_index, data.num_nodes, labels, split_idx["train"], optimizer, evaluator, struct_dev)
        val_acc = test_acc = None
        if epoch % args.eval_every == 0:
            val_acc = evaluate(model, x, edge_index, data.num_nodes, labels, split_idx["valid"], evaluator, struct_dev)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_test = evaluate(model, x, edge_index, data.num_nodes, labels, split_idx["test"], evaluator, struct_dev)
                torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
                stale = 0
            else:
                stale += args.eval_every
            test_acc = best_test
        elapsed = time.time() - t0
        append_jsonl(metrics_path, {
            "epoch": epoch, "loss": loss, "train_acc": train_acc, "val_acc": val_acc,
            "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch, "time_sec": elapsed,
        })
        logger.info(
            "epoch=%d loss=%.4f train=%.4f val=%s best_val=%.4f best_test=%.4f time=%.2fs",
            epoch, loss, train_acc, "None" if val_acc is None else f"{val_acc:.4f}",
            best_val, test_acc if test_acc is not None else best_test, elapsed,
        )
        if stale >= args.patience:
            logger.info("Early stopping at epoch=%d", epoch)
            break

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])

    result = {
        "run": run_id, "model": "sagat", "variant": args.variant,
        "num_params": count_params(model),
        "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch, "output_dir": str(out_dir),
    }

    if args.variant == "sagat" and raw_struct is not None and args.num_layers >= 2:
        correlations = analyze_structural_attention(
            model, x, edge_index, data.num_nodes, struct_dev, raw_struct, out_dir, logger
        )
        if correlations:
            result["attention_structure_correlation"] = correlations

    write_json(out_dir / "results.json", result)
    plot_path = plot_training_curves(metrics_path, out_dir, title=f"SA-GAT ({args.variant})")
    if plot_path:
        logger.info("Saved training curves: %s", plot_path)
    logger.info("Final: %s", result)
    return result


def main():
    args = parse_args()
    device = get_device(args.gpu)
    results = [run_once(args, run_id, device) for run_id in range(args.num_runs)]
    if len(results) > 1:
        tests = np.array([r["best_test"] for r in results], dtype=np.float64)
        vals = np.array([r["best_val"] for r in results], dtype=np.float64)
        print(json.dumps({
            "valid_mean": float(vals.mean()), "valid_std": float(vals.std()),
            "test_mean": float(tests.mean()), "test_std": float(tests.std()),
        }, indent=2))


if __name__ == "__main__":
    main()
