"""GAMLP (Zhang et al., 2021, "Graph Attention Multi-Layer Perceptron") on
Amazon Computers — the baseline that train_retrieval_gamlp.py's
retrieval-guided variant is compared against.

GAMLP decouples graph propagation from learning: K-hop mean-propagated node
features are precomputed once, then a purely MLP-based classifier with
learnable per-hop attention is trained on those precomputed features (no
message passing inside the training loop, so training itself is very
cheap). In "rlu" mode (Reliable Label Utilization), a teacher/student
self-training loop also propagates training labels through the graph and
feeds them in as an auxiliary signal, with high-confidence pseudo-labelled
nodes added to the training set stage by stage.

Hop-feature and label propagation here use `common.utils.propagate_mean`
(plain scatter_add) instead of the original implementation's torch-sparse
`SparseTensor.matmul` — mathematically identical on this dataset's
symmetric (undirected) adjacency, see propagate_mean's docstring — so this
script needs no compiled extension, matching every other model in this repo.
"""

import argparse
import gc
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
    append_jsonl, count_params, get_device, make_output_dir, plot_training_curves,
    propagate_mean, scatter_add, set_seed, setup_logger, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GAMLP on Amazon Computers")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/gamlp")
    parser.add_argument("--mode", choices=["plain", "rlu"], default="plain")
    parser.add_argument("--method", choices=["R_GAMLP", "JK_GAMLP"], default="R_GAMLP")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--num-hops", type=int, default=5)
    parser.add_argument("--label-num-hops", type=int, default=9)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-layers-1", type=int, default=4)
    parser.add_argument("--n-layers-2", type=int, default=4)
    parser.add_argument("--n-layers-3", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--input-drop", type=float, default=0.2)
    parser.add_argument("--att-drop", type=float, default=0.5)
    parser.add_argument("--label-drop", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--act", choices=["relu", "leaky_relu", "sigmoid"], default="leaky_relu")
    parser.add_argument("--pre-process", action="store_true", default=True)
    parser.add_argument("--no-pre-process", dest="pre_process", action="store_false")
    parser.add_argument("--residual", action="store_true", default=True)
    parser.add_argument("--no-residual", dest="residual", action="store_false")
    parser.add_argument("--bns", action="store_true", default=True)
    parser.add_argument("--no-bns", dest="bns", action="store_false")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--stages", nargs="+", type=int, default=[400, 300, 300, 300])
    parser.add_argument("--train-num-epochs", nargs="+", type=int, default=[0, 0, 0, 0])
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--gama", type=float, default=0.1)
    parser.add_argument("--split-file", type=str, default="split_idx.csv",
                        help="Pre-saved CSV split for fair cross-model comparison. "
                             "Empty string to regenerate from --seed.")

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="outputs/cache")
    return parser.parse_args()


class Dense(nn.Module):
    def __init__(self, in_features, out_features, use_bn=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.out_features ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if isinstance(self.bias, nn.BatchNorm1d):
            self.bias.reset_parameters()

    def forward(self, x):
        out = torch.mm(x, self.weight)
        out = self.bias(out)
        if self.in_features == self.out_features:
            out = out + x
        return out


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, alpha, use_bn=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.out_features ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if isinstance(self.bias, nn.BatchNorm1d):
            self.bias.reset_parameters()

    def forward(self, x, h0):
        support = (1.0 - self.alpha) * x + self.alpha * h0
        out = torch.mm(support, self.weight)
        out = self.bias(out)
        if self.in_features == self.out_features:
            out = out + x
        return out


class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, use_bn=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        self.use_bn = use_bn
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                if self.use_bn:
                    x = self.bns[layer_id](x)
                x = self.dropout(self.prelu(x))
        return x


class FeedForwardNetII(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, alpha, use_bn=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(Dense(in_feats, out_feats, use_bn=False))
        else:
            self.layers.append(Dense(in_feats, hidden, use_bn=use_bn))
            for _ in range(n_layers - 2):
                self.layers.append(GraphConvolution(hidden, hidden, alpha, use_bn=use_bn))
            self.layers.append(Dense(hidden, out_feats, use_bn=False))
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        x = self.layers[0](x)
        h0 = x
        for layer_id, layer in enumerate(self.layers[1:], start=1):
            if layer_id == self.n_layers - 1:
                x = self.dropout(self.prelu(x))
                x = layer(x)
            else:
                x = self.dropout(self.prelu(x))
                x = layer(x, h0)
        return x


def activation(name):
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    return nn.ReLU()


class RGAMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops, args, use_label=False):
        super().__init__()
        self.num_hops = num_hops
        self.pre_process = args.pre_process
        self.residual = args.residual
        self.use_label = use_label
        att_dim = hidden if self.pre_process else nfeat
        self.process = nn.ModuleList(
            [FeedForwardNet(nfeat, hidden, hidden, 2, args.dropout, args.bns) for _ in range(num_hops)]
        ) if self.pre_process else None
        self.lr_att = nn.Linear(att_dim + att_dim, 1)
        self.lr_output = FeedForwardNetII(att_dim, hidden, nclass, args.n_layers_2, args.dropout, args.alpha, args.bns)
        self.res_fc = nn.Linear(nfeat, att_dim)
        self.label_fc = FeedForwardNet(nclass, hidden, nclass, args.n_layers_3, args.dropout, args.bns) if use_label else None
        self.input_drop = nn.Dropout(args.input_drop)
        self.att_drop = nn.Dropout(args.att_drop)
        self.label_drop = nn.Dropout(args.label_drop)
        self.dropout = nn.Dropout(args.dropout)
        self.prelu = nn.PReLU()
        self.act = activation(args.act)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        if self.process is not None:
            for layer in self.process:
                layer.reset_parameters()
        if self.label_fc is not None:
            self.label_fc.reset_parameters()

    def encode(self, feature_list):
        feature_list = [self.input_drop(feature) for feature in feature_list]
        if self.pre_process:
            return [self.process[i](feature_list[i]) for i in range(self.num_hops)], feature_list
        return feature_list, feature_list

    def forward(self, feature_list, label_emb=None):
        input_list, raw_list = self.encode(feature_list)
        attention_scores = [self.act(self.lr_att(torch.cat([input_list[0], input_list[0]], dim=1)))]
        for i in range(1, self.num_hops):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, dim=1)
            history = input_list[0] * self.att_drop(att[:, 0:1])
            for j in range(1, i):
                history = history + input_list[j] * self.att_drop(att[:, j:j + 1])
            attention_scores.append(self.act(self.lr_att(torch.cat([history, input_list[i]], dim=1))))
        scores = F.softmax(torch.cat(attention_scores, dim=1), dim=1)
        hidden = input_list[0] * self.att_drop(scores[:, 0:1])
        for i in range(1, self.num_hops):
            hidden = hidden + input_list[i] * self.att_drop(scores[:, i:i + 1])
        if self.residual:
            hidden = self.dropout(self.prelu(hidden + self.res_fc(raw_list[0])))
        out = self.lr_output(hidden)
        if self.use_label and label_emb is not None:
            out = out + self.label_fc(self.label_drop(label_emb))
        return out


class JKGAMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops, args, use_label=False):
        super().__init__()
        self.num_hops = num_hops
        self.pre_process = args.pre_process
        self.residual = args.residual
        self.use_label = use_label
        att_dim = hidden if self.pre_process else nfeat
        self.process = nn.ModuleList(
            [FeedForwardNet(nfeat, hidden, hidden, 2, args.dropout, args.bns) for _ in range(num_hops)]
        ) if self.pre_process else None
        self.lr_jk_ref = FeedForwardNetII(num_hops * att_dim, hidden, hidden, args.n_layers_1, args.dropout, args.alpha, args.bns)
        self.lr_att = nn.Linear(att_dim + hidden, 1)
        self.lr_output = FeedForwardNetII(att_dim, hidden, nclass, args.n_layers_2, args.dropout, args.alpha, args.bns)
        self.res_fc = nn.Linear(nfeat, att_dim)
        self.label_fc = FeedForwardNet(nclass, hidden, nclass, args.n_layers_3, args.dropout, args.bns) if use_label else None
        self.input_drop = nn.Dropout(args.input_drop)
        self.att_drop = nn.Dropout(args.att_drop)
        self.label_drop = nn.Dropout(args.label_drop)
        self.dropout = nn.Dropout(args.dropout)
        self.prelu = nn.PReLU()
        self.act = activation(args.act)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_jk_ref.reset_parameters()
        self.lr_output.reset_parameters()
        if self.process is not None:
            for layer in self.process:
                layer.reset_parameters()
        if self.label_fc is not None:
            self.label_fc.reset_parameters()

    def forward(self, feature_list, label_emb=None):
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = [self.process[i](feature_list[i]) for i in range(self.num_hops)] if self.pre_process else feature_list
        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(torch.cat(input_list, dim=1))))
        scores = [self.act(self.lr_att(torch.cat([jk_ref, x], dim=1))) for x in input_list]
        weights = F.softmax(torch.cat(scores, dim=1), dim=1)
        hidden = input_list[0] * self.att_drop(weights[:, 0:1])
        for i in range(1, self.num_hops):
            hidden = hidden + input_list[i] * self.att_drop(weights[:, i:i + 1])
        if self.residual:
            hidden = self.dropout(self.prelu(hidden + self.res_fc(feature_list[0])))
        out = self.lr_output(hidden)
        if self.use_label and label_emb is not None:
            out = out + self.label_fc(self.label_drop(label_emb))
        return out


def build_model(args, in_feats, num_classes, use_label):
    num_hops = args.num_hops + 1
    if args.method == "JK_GAMLP":
        return JKGAMLP(in_feats, args.hidden, num_classes, num_hops, args, use_label=use_label)
    return RGAMLP(in_feats, args.hidden, num_classes, num_hops, args, use_label=use_label)


@torch.no_grad()
def precompute_features(data, num_hops, cache_dir, cache_features, logger):
    cache_name = getattr(data, "cache_name", "graph")
    cache_path = Path(cache_dir) / f"{cache_name}_hops_{num_hops}.pt"
    if cache_features and cache_path.exists():
        logger.info("Loading cached hop features from %s", cache_path)
        return torch.load(cache_path, map_location="cpu")
    logger.info("Precomputing %d-hop mean features", num_hops)
    row, col = data.edge_index
    deg = scatter_add(row.new_ones(row.size(0), 1, dtype=torch.float32), row, data.num_nodes).clamp(min=1)
    feats = [data.x.float().cpu()]
    for hop in range(1, num_hops + 1):
        start = time.time()
        feats.append(propagate_mean(data.edge_index, data.num_nodes, feats[-1], deg=deg, hops=1))
        logger.info("Computed hop %d feature in %.2fs", hop, time.time() - start)
        gc.collect()
    if cache_features:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats, cache_path)
        logger.info("Saved hop feature cache to %s", cache_path)
    return feats


@torch.no_grad()
def prepare_label_emb(edge_index, num_nodes, deg, labels, split_idx, num_classes, num_hops, teacher_probs=None):
    y = torch.zeros(labels.size(0), num_classes, dtype=torch.float32)
    train_idx = split_idx["train"]
    y[train_idx] = F.one_hot(labels[train_idx], num_classes=num_classes).float()
    if teacher_probs is not None:
        valid_idx = split_idx["valid"]
        test_idx = split_idx["test"]
        y[valid_idx] = teacher_probs[valid_idx]
        y[test_idx] = teacher_probs[test_idx]
    return propagate_mean(edge_index, num_nodes, y, deg=deg, hops=num_hops)


def ogb_acc(evaluator, y_true, y_pred):
    return evaluator.eval({"y_true": y_true.view(-1, 1), "y_pred": y_pred.view(-1, 1)})["acc"]


def run_batches(indices, batch_size, shuffle):
    return torch.utils.data.DataLoader(indices.cpu(), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_epoch(model, feats, labels, label_emb, train_idx, optimizer, evaluator, batch_size, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    y_true, y_pred = [], []
    for batch in run_batches(train_idx, batch_size, shuffle=True):
        batch_feats = [feat[batch].to(device) for feat in feats]
        batch_label_emb = label_emb[batch].to(device) if label_emb is not None else None
        out = model(batch_feats, batch_label_emb)
        y = labels[batch].to(device)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.numel()
        total_examples += batch.numel()
        y_true.append(y.detach().cpu())
        y_pred.append(out.argmax(dim=-1).detach().cpu())
    acc = ogb_acc(evaluator, torch.cat(y_true), torch.cat(y_pred))
    return total_loss / max(total_examples, 1), acc


def train_epoch_rlu(model, feats, labels, label_emb, train_idx, enhance_idx, teacher_probs, optimizer, evaluator, args, device):
    model.train()
    train_loader = run_batches(train_idx, max(1, int(args.batch_size * len(train_idx) / max(len(train_idx) + len(enhance_idx), 1))), True)
    enhance_loader = run_batches(enhance_idx, max(1, int(args.batch_size * len(enhance_idx) / max(len(train_idx) + len(enhance_idx), 1))), True)
    loss_sum = 0.0
    total_examples = 0
    y_true, y_pred = [], []
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat([idx_1, idx_2], dim=0)
        batch_feats = [feat[idx].to(device) for feat in feats]
        out = model(batch_feats, label_emb[idx].to(device))
        hard_loss = F.cross_entropy(out[: idx_1.numel()], labels[idx_1].to(device))
        teacher_soft = teacher_probs[idx_2].to(device)
        teacher_conf = teacher_soft.max(dim=1, keepdim=True).values
        kl_loss = (
            teacher_conf
            * (teacher_soft * (torch.log(teacher_soft + 1e-8) - F.log_softmax(out[idx_1.numel():], dim=1))).sum(dim=1, keepdim=True)
        ).mean()
        ratio_hard = idx_1.numel() / max(idx.numel(), 1)
        ratio_soft = idx_2.numel() / max(idx.numel(), 1)
        loss = hard_loss * ratio_hard + kl_loss * ratio_soft * args.gama
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * idx.numel()
        total_examples += idx.numel()
        y_true.append(labels[idx_1].cpu())
        y_pred.append(out[: idx_1.numel()].argmax(dim=-1).detach().cpu())
    acc = ogb_acc(evaluator, torch.cat(y_true), torch.cat(y_pred))
    return loss_sum / max(total_examples, 1), acc


@torch.no_grad()
def evaluate(model, feats, labels, label_emb, idx, evaluator, batch_size, device):
    model.eval()
    preds = []
    for batch in run_batches(idx, batch_size, shuffle=False):
        batch_feats = [feat[batch].to(device) for feat in feats]
        batch_label_emb = label_emb[batch].to(device) if label_emb is not None else None
        preds.append(model(batch_feats, batch_label_emb).argmax(dim=-1).cpu())
    pred = torch.cat(preds)
    return ogb_acc(evaluator, labels[idx.cpu()], pred)


@torch.no_grad()
def predict_logits(model, feats, label_emb, batch_size, device):
    model.eval()
    logits = []
    all_idx = torch.arange(feats[0].size(0))
    for batch in run_batches(all_idx, batch_size, shuffle=False):
        batch_feats = [feat[batch].to(device) for feat in feats]
        batch_label_emb = label_emb[batch].to(device) if label_emb is not None else None
        logits.append(model(batch_feats, batch_label_emb).cpu())
    return torch.cat(logits, dim=0)


def train_stage(args, stage, epochs, model, feats, labels, label_emb, split_idx, evaluator, out_dir, logger, device, teacher_probs=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    stale = 0
    train_idx = split_idx["train"]
    enhance_idx = None
    if args.mode == "rlu" and stage > 0 and teacher_probs is not None:
        conf = teacher_probs.max(dim=1).values
        mask = conf > args.threshold
        mask[train_idx] = False
        enhance_idx = torch.nonzero(mask, as_tuple=False).view(-1).cpu()
        logger.info("Stage %d confident extra nodes: %d", stage, enhance_idx.numel())
        if enhance_idx.numel() == 0:
            enhance_idx = train_idx.cpu()

    ckpt_path = out_dir / f"best_stage_{stage}.pt"
    metrics_path = out_dir / "metrics.jsonl"
    for epoch in range(epochs):
        start = time.time()
        if args.mode == "rlu" and stage > 0 and teacher_probs is not None:
            loss, train_acc = train_epoch_rlu(model, feats, labels, label_emb, train_idx, enhance_idx, teacher_probs, optimizer, evaluator, args, device)
        else:
            loss, train_acc = train_epoch(model, feats, labels, label_emb, train_idx, optimizer, evaluator, args.batch_size, device)
        val_acc = None
        test_acc = None
        if epoch % args.eval_every == 0 and epoch >= args.train_num_epochs[min(stage, len(args.train_num_epochs) - 1)]:
            val_acc = evaluate(model, feats, labels, label_emb, split_idx["valid"], evaluator, args.batch_size, device)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_test = evaluate(model, feats, labels, label_emb, split_idx["test"], evaluator, args.batch_size, device)
                torch.save({"model_state": model.state_dict(), "args": vars(args), "stage": stage}, ckpt_path)
                stale = 0
            else:
                stale += args.eval_every
            test_acc = best_test
        elapsed = time.time() - start
        payload = {
            "stage": stage, "epoch": epoch, "loss": loss, "train_acc": train_acc,
            "val_acc": val_acc, "best_val": best_val, "best_test": best_test,
            "best_epoch": best_epoch, "time_sec": elapsed,
        }
        append_jsonl(metrics_path, payload)
        logger.info(
            "stage=%d epoch=%d loss=%.4f train=%.4f val=%s best_val=%.4f best_test=%.4f time=%.2fs",
            stage, epoch, loss, train_acc,
            "None" if val_acc is None else f"{val_acc:.4f}",
            best_val, test_acc if test_acc is not None else best_test, elapsed,
        )
        if stale >= args.patience:
            logger.info("Early stopping at stage=%d epoch=%d", stage, epoch)
            break
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    return best_val, best_test, best_epoch, ckpt_path


def run_once(args, run_id, device):
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}_{args.mode}_{args.method}"
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger("gamlp", out_dir)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))
    set_seed(args.seed + run_id)

    data, labels, split_idx, num_classes = load_products(args.dataset_root, logger, split_seed=args.seed + run_id)
    split_file = Path(args.split_file) if args.split_file else None
    if split_file and split_file.is_file():
        split_idx = load_split_idx_csv(split_file)
        logger.info("Fixed split from %s  train=%d valid=%d test=%d",
                    split_file, split_idx["train"].numel(), split_idx["valid"].numel(), split_idx["test"].numel())

    feats = precompute_features(data, args.num_hops, args.cache_dir, args.cache_features, logger)
    in_feats = feats[0].size(1)
    evaluator = AccuracyEvaluator()
    row, col = data.edge_index
    deg = scatter_add(row.new_ones(row.size(0), 1, dtype=torch.float32), row, data.num_nodes).clamp(min=1)
    labels = labels.cpu()
    split_idx = {k: v.cpu() for k, v in split_idx.items()}

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        use_label = "rlu" in args.checkpoint.lower() or args.mode == "rlu"
        label_emb = prepare_label_emb(data.edge_index, data.num_nodes, deg, labels, split_idx, num_classes, args.label_num_hops) if use_label else None
        model = build_model(args, in_feats, num_classes, use_label=use_label).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        result = {
            "train_acc": evaluate(model, feats, labels, label_emb, split_idx["train"], evaluator, args.batch_size, device),
            "valid_acc": evaluate(model, feats, labels, label_emb, split_idx["valid"], evaluator, args.batch_size, device),
            "test_acc": evaluate(model, feats, labels, label_emb, split_idx["test"], evaluator, args.batch_size, device),
        }
        write_json(out_dir / "eval_results.json", result)
        logger.info("Eval-only results: %s", result)
        return result

    teacher_probs = None
    stage_results = []
    stages = args.stages if args.mode == "rlu" else [args.epochs]
    for stage, epochs in enumerate(stages):
        label_emb = None
        use_label = args.mode == "rlu"
        if use_label:
            label_emb = prepare_label_emb(data.edge_index, data.num_nodes, deg, labels, split_idx, num_classes, args.label_num_hops, teacher_probs)
        model = build_model(args, in_feats, num_classes, use_label=use_label).to(device)
        logger.info("Stage %d model params: %d", stage, count_params(model))
        best_val, best_test, best_epoch, ckpt_path = train_stage(
            args, stage, epochs, model, feats, labels, label_emb, split_idx, evaluator, out_dir, logger, device, teacher_probs
        )
        logits = predict_logits(model, feats, label_emb, args.batch_size, device)
        torch.save(logits, out_dir / f"logits_stage_{stage}.pt")
        np.save(out_dir / f"logits_stage_{stage}.npy", logits.numpy())
        teacher_probs = (logits / args.temp).softmax(dim=1)
        stage_results.append(
            {"stage": stage, "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch, "checkpoint": str(ckpt_path)}
        )
        gc.collect()

    result = {
        "run": run_id, "mode": args.mode, "method": args.method,
        "num_params": count_params(model),  # last stage's model
        "best_val": stage_results[-1]["best_val"], "best_test": stage_results[-1]["best_test"],
        "stages": stage_results, "output_dir": str(out_dir),
    }
    write_json(out_dir / "results.json", result)
    plot_path = plot_training_curves(out_dir / "metrics.jsonl", out_dir, title=f"GAMLP ({args.mode}, {args.method})")
    if plot_path:
        logger.info("Saved training curves: %s", plot_path)
    logger.info("Final result: %s", result)
    return result


def main():
    args = parse_args()
    device = get_device(args.gpu)
    results = [run_once(args, run_id, device) for run_id in range(args.num_runs)]
    if len(results) > 1 and "best_test" in results[0]:
        tests = np.array([r["best_test"] for r in results], dtype=np.float64)
        vals = np.array([r["best_val"] for r in results], dtype=np.float64)
        print(json.dumps({
            "valid_mean": float(vals.mean()), "valid_std": float(vals.std()),
            "test_mean": float(tests.mean()), "test_std": float(tests.std()),
        }, indent=2))


if __name__ == "__main__":
    main()
