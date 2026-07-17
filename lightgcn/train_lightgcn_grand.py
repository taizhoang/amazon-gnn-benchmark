"""LightGCN-GRAND on Amazon Computers.

Combines two ideas:

1. **LightGCN** (He et al., 2020) — propagation with no per-layer linear
   transform or nonlinearity between hops. A single feature/logit vector is
   propagated K times through the normalised adjacency, and the layer
   outputs are combined with a learned (softmax) weight per hop, instead of
   the usual "transform then propagate then activate" GCN layer stack.
2. **GRAND** (Feng et al., 2020) — DropNode augmentation (randomly zeroing
   whole node feature vectors, rather than individual entries like standard
   dropout) plus a consistency-regularisation loss: several DropNode-perturbed
   forward passes should agree with each other, sharpened towards their own
   average prediction, only on nodes the model is already confident about.

A third piece, kept optional via `--use-H`: a **class-Compatibility
Propagation (CoP) channel**. A small SGC + logistic-regression "teacher" is
fit once on the training labels to produce pseudo-labels for every node;
counting how often each pair of classes co-occurs across edges gives a
C x C compatibility matrix, which seeds a second, learnable propagation
channel that reweights the graph-propagated logits.

This is a from-scratch pure-PyTorch port of a TensorFlow prototype
(`lightGCN-GRAND.py` at the repo root) — model behaviour is unchanged, but
propagation uses this repo's own scatter-based ops (`common/utils.py`)
instead of `tf.sparse`, and dataset loading / the fixed evaluation split are
now the same `common/data.py` + `split_idx.csv` every other model in this
benchmark uses (see `--low-label-split` below for the original alternative).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.data import AccuracyEvaluator, load_products, load_split_idx_csv
from common.utils import (
    add_self_loops, append_jsonl, count_params, gcn_norm_edge_weight, get_device,
    make_output_dir, plot_training_curves, set_seed, setup_logger, weighted_propagate,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="LightGCN-GRAND on Amazon Computers")
    p.add_argument("--dataset-root", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="outputs/lightgcn_grand")
    p.add_argument("--num-runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--K", type=int, default=4, help="propagation hops (LightGCN layer combination)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropnode", type=float, default=0.5, help="GRAND DropNode rate")
    p.add_argument("--dropout", type=float, default=0.5, help="standard dropout in the input MLP")
    p.add_argument("--S", type=int, default=2, help="DropNode augmentations per training step")
    p.add_argument("--lam", type=float, default=1.0, help="consistency loss weight after warmup")
    p.add_argument("--temp", type=float, default=0.5, help="sharpening temperature for the consistency target")
    p.add_argument("--conf", type=float, default=0.7, help="confidence threshold gating the consistency loss")
    p.add_argument("--warmup", type=float, default=60.0, help="epochs to linearly warm lambda up to --lam")
    p.add_argument("--prop-space", choices=["logits", "hidden"], default="logits",
                   help="propagate class logits (cheap, APPNP/GPR-GNN-style) or the 64-d hidden layer (closer to vanilla GRAND)")
    p.add_argument("--no-learn-gamma", dest="learn_gamma", action="store_false",
                   help="fix layer-combination weights to uniform instead of learning them")
    p.set_defaults(learn_gamma=True)

    p.add_argument("--no-use-H", dest="use_H", action="store_false",
                   help="disable the class-Compatibility-Propagation (CoP) channel")
    p.set_defaults(use_H=True)
    p.add_argument("--Kc", type=int, default=2, help="propagation hops inside the CoP channel")

    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--l2-reg", type=float, default=5e-4,
                   help="L2 penalty on the two linear layers, added directly to the loss (not the optimizer's weight_decay)")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--min-epochs", type=int, default=120, help="no early stopping before this epoch")
    p.add_argument("--split-file", type=str, default="split_idx.csv",
                   help="Pre-saved CSV split for fair cross-model comparison. "
                        "Empty string to regenerate from --seed.")
    p.add_argument("--low-label-split", action="store_true",
                   help="Use the original paper-style N-train/M-valid-per-class split instead of "
                        "--split-file, to reproduce GRAND's low-label-rate regime. Not comparable "
                        "to other models' results.json in this benchmark, which all use the fixed split.")
    p.add_argument("--low-label-train-per-class", type=int, default=20)
    p.add_argument("--low-label-valid-per-class", type=int, default=30)
    return p.parse_args()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class LightGCNGRAND(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, K, learn_gamma=True, use_H=True, Kc=2, H_init=None):
        super().__init__()
        self.K, self.use_H, self.Kc = K, use_H, Kc
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.gamma = nn.Parameter(torch.zeros(K + 1), requires_grad=learn_gamma)
        if use_H:
            assert H_init is not None
            self.H = nn.Parameter(H_init.clone())
            self.delta = nn.Parameter(torch.zeros(Kc))
            self.mix = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ~ 0.12: CoP starts as a small correction

    def propagate(self, x, edge_index, edge_weight, num_nodes):
        """Core LightGCN: no transform, no nonlinearity between hops — just a
        learned (softmax) weighted sum of the K propagated versions of x."""
        g = F.softmax(self.gamma, dim=0)
        ego = x
        out = g[0] * ego
        for k in range(1, self.K + 1):
            ego = weighted_propagate(ego, edge_index, edge_weight, num_nodes)
            out = out + g[k] * ego
        return out

    def forward(self, x, edge_index, edge_weight, num_nodes, training, dropnode, dropout, prop_space):
        if training and dropnode > 0:
            # DropNode (GRAND): zero a node's entire feature vector, not individual
            # entries, then rescale by 1/(1-p) to keep the expectation unchanged.
            mask = (torch.rand(num_nodes, 1, device=x.device) > dropnode).float() / (1.0 - dropnode)
            x = x * mask
        h = F.relu(self.lin1(x))
        if training:
            h = F.dropout(h, dropout, training=True)
        if prop_space == "hidden":
            h = self.propagate(h, edge_index, edge_weight, num_nodes)
            return self.lin2(h)
        # Default: MLP first, then propagate the low-dim LOGITS (APPNP/GPR-GNN
        # order) — far cheaper than propagating the hidden layer while keeping
        # the same LightGCN core (no transform between propagation hops).
        logits0 = self.lin2(h)
        zA = self.propagate(logits0, edge_index, edge_weight, num_nodes)
        if not self.use_H:
            return zA
        # CoP channel: push the node-level class distribution through a learned
        # class-compatibility matrix at every hop, propagate that through the
        # graph too, and blend it in on a log scale.
        Hn = F.softmax(self.H, dim=1)
        P = F.softmax(logits0, dim=1)
        dl = F.softmax(self.delta, dim=0)
        ego, zB = P, torch.zeros_like(P)
        for k in range(self.Kc):
            ego = weighted_propagate(ego @ Hn, edge_index, edge_weight, num_nodes)
            zB = zB + dl[k] * ego
        m = torch.sigmoid(self.mix)
        return zA + m * torch.log(zB + 1e-8)


# ----------------------------------------------------------------------------
# CoP teacher: SGC + logistic regression pseudo-labels -> class compatibility matrix
# ----------------------------------------------------------------------------

def build_normalized_adjacency_scipy(edge_index, num_nodes):
    row, col = edge_index.numpy()
    adj = sp.csr_matrix((np.ones(row.shape[0], dtype=np.float32), (row, col)), shape=(num_nodes, num_nodes))
    adj = adj.maximum(adj.T)
    adj.data[:] = 1
    m = adj + sp.eye(num_nodes)
    deg = np.asarray(m.sum(1)).ravel()
    d_inv_sqrt = sp.diags(deg ** -0.5)
    return (d_inv_sqrt @ m @ d_inv_sqrt).tocsr()


def teacher_h_init(x_np, labels_np, train_idx, edge_index, num_classes, mn_sp, seed):
    hk = x_np.copy()
    acc = x_np.copy()
    for _ in range(2):
        hk = mn_sp @ hk
        acc = acc + hk
    feat_teacher = acc / 3.0
    clf = LogisticRegression(max_iter=1500, C=10.0, random_state=seed).fit(feat_teacher[train_idx], labels_np[train_idx])
    pseudo_labels = clf.predict(feat_teacher)

    row, col = edge_index.numpy()
    unique_edges = row < col  # edge_index is symmetric, so this dedupes to one row per undirected edge
    e_row, e_col = row[unique_edges], col[unique_edges]
    pairs = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(pairs, (pseudo_labels[e_row], pseudo_labels[e_col]), 1.0)
    pairs = pairs + pairs.T
    pairs = pairs / pairs.sum(1, keepdims=True).clip(min=1e-8)
    return torch.from_numpy(pairs.astype(np.float32))


def make_low_label_split(labels_np, num_classes, seed, n_train, n_valid):
    rng = np.random.RandomState(seed)
    train, valid, test = [], [], []
    for c in range(num_classes):
        idx = rng.permutation(np.where(labels_np == c)[0])
        train += list(idx[:n_train])
        valid += list(idx[n_train:n_train + n_valid])
        test += list(idx[n_train + n_valid:])
    return {
        "train": torch.tensor(train, dtype=torch.long),
        "valid": torch.tensor(valid, dtype=torch.long),
        "test": torch.tensor(test, dtype=torch.long),
    }


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------

def forward_pass(model, x, edge_index, edge_weight, num_nodes, training, args):
    return model(x, edge_index, edge_weight, num_nodes, training=training,
                dropnode=args.dropnode, dropout=args.dropout, prop_space=args.prop_space)


def train_epoch(model, x, edge_index, edge_weight, num_nodes, labels, train_idx, optimizer, evaluator, lam, args):
    model.train()
    probs = []
    for _ in range(args.S):
        logits = forward_pass(model, x, edge_index, edge_weight, num_nodes, True, args)
        probs.append(F.softmax(logits, dim=1))

    sup = sum(F.nll_loss(torch.log(pr[train_idx] + 1e-8), labels[train_idx]) for pr in probs) / args.S

    avg = sum(probs) / args.S
    sharp = avg.pow(1.0 / args.temp)
    sharp = (sharp / sharp.sum(dim=1, keepdim=True)).detach()
    conf_mask = (avg.max(dim=1).values > args.conf).float().detach()
    denom = conf_mask.sum() + 1e-8
    con = sum((conf_mask * (pr - sharp).pow(2).sum(dim=1)).sum() / denom for pr in probs) / args.S

    l2 = args.l2_reg * 0.5 * (model.lin1.weight.pow(2).sum() + model.lin2.weight.pow(2).sum())
    loss = sup + lam * con + l2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred = avg[train_idx].argmax(dim=1).cpu()
        acc = evaluator.eval({"y_true": labels[train_idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]
    return float(loss.item()), acc


@torch.no_grad()
def evaluate(model, x, edge_index, edge_weight, num_nodes, labels, idx, evaluator, args):
    model.eval()
    logits = forward_pass(model, x, edge_index, edge_weight, num_nodes, False, args)
    pred = logits[idx].argmax(dim=1).cpu()
    return evaluator.eval({"y_true": labels[idx].cpu().view(-1, 1), "y_pred": pred.view(-1, 1)})["acc"]


def run_once(args, run_id, device):
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}_lightgcn_grand"
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger("lightgcn_grand", out_dir)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))
    set_seed(args.seed + run_id)

    data, labels, split_idx, num_classes = load_products(args.dataset_root, logger, split_seed=args.seed)
    if args.low_label_split:
        split_idx = make_low_label_split(
            labels.numpy(), num_classes, args.seed + run_id,
            args.low_label_train_per_class, args.low_label_valid_per_class,
        )
        logger.info("Low-label split (paper-style)  train=%d valid=%d test=%d",
                    split_idx["train"].numel(), split_idx["valid"].numel(), split_idx["test"].numel())
    else:
        split_file = Path(args.split_file) if args.split_file else None
        if split_file and split_file.is_file():
            split_idx = load_split_idx_csv(split_file)
            logger.info("Fixed split from %s  train=%d valid=%d test=%d",
                        split_file, split_idx["train"].numel(), split_idx["valid"].numel(), split_idx["test"].numel())

    x = data.x.float()
    edge_index_sl = add_self_loops(data.edge_index, data.num_nodes)
    edge_weight = gcn_norm_edge_weight(edge_index_sl, data.num_nodes)

    H_init = None
    if args.use_H:
        mn_sp = build_normalized_adjacency_scipy(data.edge_index, data.num_nodes)
        H_init = teacher_h_init(
            x.numpy(), labels.numpy(), split_idx["train"].numpy(),
            data.edge_index, num_classes, mn_sp, args.seed + run_id,
        ).to(device)

    x, edge_index_sl, edge_weight = x.to(device), edge_index_sl.to(device), edge_weight.to(device)
    labels_dev = labels.to(device)
    split_dev = {k: v.to(device) for k, v in split_idx.items()}
    evaluator = AccuracyEvaluator()

    model = LightGCNGRAND(
        x.size(1), args.hidden, num_classes, args.K,
        learn_gamma=args.learn_gamma, use_H=args.use_H, Kc=args.Kc, H_init=H_init,
    ).to(device)
    logger.info("Model params: %d", count_params(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = best_test = -1.0
    best_epoch = stale = 0
    ckpt_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.jsonl"

    for epoch in range(args.epochs):
        t0 = time.time()
        lam = args.lam * min(1.0, epoch / args.warmup) if args.warmup > 0 else args.lam
        loss, train_acc = train_epoch(
            model, x, edge_index_sl, edge_weight, data.num_nodes, labels_dev, split_dev["train"], optimizer, evaluator, lam, args
        )
        val_acc = evaluate(model, x, edge_index_sl, edge_weight, data.num_nodes, labels_dev, split_dev["valid"], evaluator, args)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_test = evaluate(model, x, edge_index_sl, edge_weight, data.num_nodes, labels_dev, split_dev["test"], evaluator, args)
            torch.save({"model_state": model.state_dict()}, ckpt_path)
            stale = 0
        else:
            stale += 1
        elapsed = time.time() - t0
        append_jsonl(metrics_path, {
            "epoch": epoch, "loss": loss, "train_acc": train_acc, "val_acc": val_acc,
            "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch,
            "lambda": lam, "time_sec": elapsed,
        })
        logger.info(
            "epoch=%d loss=%.4f train=%.4f val=%.4f best_val=%.4f best_test=%.4f lam=%.3f time=%.2fs",
            epoch, loss, train_acc, val_acc, best_val, best_test, lam, elapsed,
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            logger.info("Early stopping at epoch=%d", epoch)
            break

    result = {
        "run": run_id, "model": "lightgcn_grand",
        "prop_space": args.prop_space, "use_H": args.use_H, "learn_gamma": args.learn_gamma,
        "num_params": count_params(model),
        "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch, "output_dir": str(out_dir),
    }
    write_json(out_dir / "results.json", result)
    plot_path = plot_training_curves(metrics_path, out_dir, title="LightGCN-GRAND")
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
