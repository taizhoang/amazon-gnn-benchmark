"""GraphSAGE baseline (Hamilton et al., 2017, mean aggregator) on Amazon Computers.

Layer: h_v' = act(W_self * h_v + W_neigh * mean_{u in N(v)} h_u)

Two training modes, selected by `--sampling`:
  full (default)  — whole graph in one forward/backward pass. Fine here because
                     Amazon Computers is small (~13.7K nodes); peak memory stays
                     well under 200MB.
  sampled         — mini-batch neighbour sampling, GraphSAGE's actual claim to
                     fame and the standard way to keep memory bounded on graphs
                     that don't fit in memory: each step draws a batch of seed
                     nodes, expands a fixed-size (`--fanout`) random neighbour
                     set per hop, builds the small induced subgraph, and runs
                     the 2-layer model only on those nodes. Peak memory is
                     O(batch_size * fanout^num_layers) instead of O(N + E).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.data import AccuracyEvaluator, load_products, load_split_idx_csv
from common.utils import (
    append_jsonl, build_adjacency_list, count_params, get_device, make_output_dir,
    plot_training_curves, scatter_add, set_seed, setup_logger, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GraphSAGE (mean) baseline on Amazon Computers")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/graphsage")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1)

    # Mini-batch neighbour sampling (the resource-constrained mode)
    parser.add_argument("--sampling", action="store_true", help="use mini-batch neighbour sampling instead of full-batch")
    parser.add_argument("--batch-size", type=int, default=512, help="seed nodes per mini-batch when --sampling is set")
    parser.add_argument("--fanout", type=int, nargs="+", default=[10, 10], help="neighbours sampled per hop, outer-most hop first, one value per layer")
    parser.add_argument("--split-file", type=str, default="split_idx.csv",
                        help="Pre-saved CSV split for fair cross-model comparison. "
                             "Empty string to regenerate from --seed.")
    return parser.parse_args()


class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index
        neigh_sum = scatter_add(x[col], row, num_nodes)
        deg = scatter_add(x.new_ones(col.size(0), 1), row, num_nodes).clamp(min=1)
        neigh_mean = neigh_sum / deg
        return self.lin_self(x) + self.lin_neigh(neigh_mean)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers, dropout):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([SAGELayer(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, num_nodes):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, num_nodes)
            x = self.dropout(F.relu(self.bns[i](x)))
        return self.convs[-1](x, edge_index, num_nodes)


# ----------------------------------------------------------------------------
# Full-batch training
# ----------------------------------------------------------------------------

def train_epoch_full(model, x, edge_index, num_nodes, labels, train_idx, optimizer, evaluator):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, num_nodes)
    loss = F.cross_entropy(out[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    pred = out[train_idx].argmax(dim=-1).detach().cpu()
    acc = evaluator.eval({"y_true": labels[train_idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]
    return float(loss.item()), acc


@torch.no_grad()
def evaluate_full(model, x, edge_index, num_nodes, labels, idx, evaluator):
    model.eval()
    out = model(x, edge_index, num_nodes)
    pred = out[idx].argmax(dim=-1).cpu()
    return evaluator.eval({"y_true": labels[idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]


# ----------------------------------------------------------------------------
# Mini-batch neighbour sampling
# ----------------------------------------------------------------------------

def sample_subgraph(adj_list, seed_nodes, fanouts):
    """BFS-style ego-graph sampling: expands `seed_nodes` outward for len(fanouts)
    hops, sampling up to `fanout` random neighbours per node at each hop, and
    returns the induced (locally re-indexed) subgraph edge_index plus a
    mapping from local index back to the original global node id."""
    frontier = set(seed_nodes.tolist())
    visited = set(frontier)
    for fanout in fanouts:
        next_frontier = set()
        for n in frontier:
            neighbours = adj_list[n]
            if neighbours.numel() == 0:
                continue
            if neighbours.numel() > fanout:
                sampled = neighbours[torch.randperm(neighbours.numel())[:fanout]]
            else:
                sampled = neighbours
            next_frontier.update(sampled.tolist())
        visited.update(next_frontier)
        frontier = next_frontier

    node_ids = torch.tensor(sorted(visited), dtype=torch.long)
    global_to_local = {int(g): i for i, g in enumerate(node_ids.tolist())}
    node_set = visited
    rows, cols = [], []
    for n in node_ids.tolist():
        for nb in adj_list[n].tolist():
            if nb in node_set:
                rows.append(global_to_local[n])
                cols.append(global_to_local[nb])
    if not rows:
        rows, cols = list(global_to_local.values()), list(global_to_local.values())
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return node_ids, edge_index


def run_batches(indices, batch_size, shuffle):
    return torch.utils.data.DataLoader(indices.cpu(), batch_size=batch_size, shuffle=shuffle)


def train_epoch_sampled(model, x, adj_list, labels, train_idx, fanouts, batch_size, optimizer, evaluator, device):
    model.train()
    total_loss = total_n = 0
    y_true, y_pred = [], []
    for seed in run_batches(train_idx, batch_size, shuffle=True):
        node_ids, sub_edge_index = sample_subgraph(adj_list, seed, fanouts)
        sub_x = x[node_ids].to(device)
        sub_edge_index = sub_edge_index.to(device)
        out = model(sub_x, sub_edge_index, node_ids.size(0))
        local_seed = torch.searchsorted(node_ids, seed)  # node_ids is sorted, so this recovers seed positions
        y = labels[seed].to(device)
        loss = F.cross_entropy(out[local_seed], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * seed.numel()
        total_n += seed.numel()
        y_true.append(y.cpu())
        y_pred.append(out[local_seed].argmax(dim=-1).detach().cpu())
    acc = evaluator.eval({"y_true": torch.cat(y_true).view(-1, 1), "y_pred": torch.cat(y_pred).view(-1, 1)})["acc"]
    return total_loss / max(total_n, 1), acc


@torch.no_grad()
def evaluate_sampled(model, x, edge_index, num_nodes, labels, idx, evaluator, device):
    # Evaluation still runs full-batch: inference is a single forward pass (no
    # backward graph to keep around), so it's cheap even though training used
    # sampling to bound the *training* memory footprint.
    model.eval()
    out = model(x.to(device), edge_index.to(device), num_nodes)
    pred = out[idx].argmax(dim=-1).cpu()
    return evaluator.eval({"y_true": labels[idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]


def run_once(args, run_id, device):
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}_graphsage_{'sampled' if args.sampling else 'full'}"
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger("graphsage", out_dir)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))
    set_seed(args.seed + run_id)

    data, labels, split_idx, num_classes = load_products(args.dataset_root, logger, split_seed=args.seed)
    split_file = Path(args.split_file) if args.split_file else None
    if split_file and split_file.is_file():
        split_idx = load_split_idx_csv(split_file)
        logger.info("Fixed split from %s  train=%d valid=%d test=%d",
                    split_file, split_idx["train"].numel(), split_idx["valid"].numel(), split_idx["test"].numel())
    x = data.x.float()
    edge_index = data.edge_index
    labels_cpu = labels.cpu()
    split_idx_cpu = {k: v.cpu() for k, v in split_idx.items()}
    evaluator = AccuracyEvaluator()

    model = GraphSAGE(x.size(1), args.hidden, num_classes, args.num_layers, args.dropout).to(device)
    logger.info("Model params: %d", count_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.sampling:
        fanouts = (args.fanout * args.num_layers)[:args.num_layers]
        adj_list = build_adjacency_list(edge_index, data.num_nodes)
        logger.info("Mini-batch neighbour sampling: fanouts=%s batch_size=%d", fanouts, args.batch_size)
    else:
        x = x.to(device)
        edge_index = edge_index.to(device)
        labels_gpu = labels.to(device)
        split_idx_gpu = {k: v.to(device) for k, v in split_idx.items()}

    best_val = best_test = -1.0
    best_epoch = stale = 0
    ckpt_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.jsonl"

    for epoch in range(args.epochs):
        t0 = time.time()
        if args.sampling:
            loss, train_acc = train_epoch_sampled(
                model, x, adj_list, labels_cpu, split_idx_cpu["train"], fanouts, args.batch_size, optimizer, evaluator, device
            )
        else:
            loss, train_acc = train_epoch_full(model, x, edge_index, data.num_nodes, labels_gpu, split_idx_gpu["train"], optimizer, evaluator)

        val_acc = test_acc = None
        if epoch % args.eval_every == 0:
            if args.sampling:
                val_acc = evaluate_sampled(model, x, edge_index, data.num_nodes, labels_cpu, split_idx_cpu["valid"], evaluator, device)
            else:
                val_acc = evaluate_full(model, x, edge_index, data.num_nodes, labels_gpu, split_idx_gpu["valid"], evaluator)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                if args.sampling:
                    best_test = evaluate_sampled(model, x, edge_index, data.num_nodes, labels_cpu, split_idx_cpu["test"], evaluator, device)
                else:
                    best_test = evaluate_full(model, x, edge_index, data.num_nodes, labels_gpu, split_idx_gpu["test"], evaluator)
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

    result = {
        "run": run_id, "model": "graphsage", "mode": "sampled" if args.sampling else "full",
        "num_params": count_params(model),
        "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch, "output_dir": str(out_dir),
    }
    write_json(out_dir / "results.json", result)
    plot_path = plot_training_curves(
        metrics_path, out_dir,
        title=f"GraphSAGE ({'sampled' if args.sampling else 'full-batch'})",
    )
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
