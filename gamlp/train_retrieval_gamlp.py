"""Retrieval-Guided Adaptive Multi-Hop GAMLP on Amazon Computers.

Extends the plain GAMLP baseline (train_gamlp.py) with a second, independent
source of neighbour information: for every node, the top-k nodes most
similar in *feature* space (cosine similarity over averaged hop features,
found via FAISS or an exact torch fallback) are retrieved and their features
aggregated the same way graph-hop features are. A learned gate per hop then
blends the graph-propagated feature with the retrieval-aggregated one before
attention pools across hops.

Motivation: graph-hop propagation only ever sees a node's *topological*
neighbours. On a co-purchase graph that's usually the right signal, but it
misses nodes that are similar in content/feature space yet not directly
connected (e.g. two accessories for the same product line that were never
co-purchased together). Retrieval augmentation adds that missing signal
without changing the graph structure itself.

Like train_gamlp.py, hop-feature propagation here uses
`common.utils.propagate_mean` (plain scatter_add) instead of the original
prototype's torch-sparse `SparseTensor.matmul` — see that function's
docstring for why this is exact, not approximate, on this dataset.
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
    parser = argparse.ArgumentParser(
        description="Retrieval-Guided Adaptive Multi-Hop GAMLP on Amazon Computers"
    )
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/retrieval_gamlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--num-hops", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-layers-1", type=int, default=4)
    parser.add_argument("--n-layers-2", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--input-drop", type=float, default=0.2)
    parser.add_argument("--att-drop", type=float, default=0.5)
    parser.add_argument("--retrieval-drop", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--act", choices=["relu", "leaky_relu", "sigmoid"], default="leaky_relu")
    parser.add_argument("--pre-process", action="store_true", default=True)
    parser.add_argument("--no-pre-process", dest="pre_process", action="store_false")
    parser.add_argument("--residual", action="store_true", default=True)
    parser.add_argument("--no-residual", dest="residual", action="store_false")
    parser.add_argument("--bns", action="store_true", default=True)
    parser.add_argument("--no-bns", dest="bns", action="store_false")

    parser.add_argument("--retrieval-topk", type=int, default=16)
    parser.add_argument("--retrieval-backend", choices=["auto", "faiss", "torch"], default="auto")
    parser.add_argument("--retrieval-embedding", choices=["hop0", "mean_hops"], default="mean_hops")
    parser.add_argument("--retrieval-temp", type=float, default=0.07)
    parser.add_argument("--retrieval-chunk-size", type=int, default=256)
    parser.add_argument("--faiss-index", choices=["hnsw", "flat"], default="hnsw")
    parser.add_argument("--faiss-hnsw-m", type=int, default=32)
    parser.add_argument("--faiss-ef-search", type=int, default=128)
    parser.add_argument("--aggregation-chunk-size", type=int, default=200000)
    parser.add_argument("--cache-dir", type=str, default="outputs/cache")
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--cache-retrieval", action="store_true")
    parser.add_argument("--cache-dtype", choices=["float32", "float16"], default="float16")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--split-file", type=str, default="split_idx.csv",
                        help="Pre-saved CSV split for fair cross-model comparison. "
                             "Empty string to regenerate from --seed.")

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
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


class RetrievalGAMLP(nn.Module):
    def __init__(self, nfeat, hidden, nclass, num_hops, args):
        super().__init__()
        self.num_hops = num_hops
        self.pre_process = args.pre_process
        self.residual = args.residual
        self.input_drop = nn.Dropout(args.input_drop)
        self.att_drop = nn.Dropout(args.att_drop)
        self.retrieval_drop = nn.Dropout(args.retrieval_drop)
        self.dropout = nn.Dropout(args.dropout)
        self.prelu = nn.PReLU()
        self.act = activation(args.act)

        dim = hidden if self.pre_process else nfeat
        if self.pre_process:
            self.graph_process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, args.dropout, args.bns) for _ in range(num_hops)]
            )
            self.ret_process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, args.dropout, args.bns) for _ in range(num_hops)]
            )
        else:
            self.graph_process = None
            self.ret_process = None

        self.gates = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(num_hops)])
        self.lr_jk_ref = FeedForwardNetII(num_hops * dim, hidden, hidden, args.n_layers_1, args.dropout, args.alpha, args.bns)
        self.lr_att = nn.Linear(dim + hidden, 1)
        self.lr_output = FeedForwardNetII(dim, hidden, nclass, args.n_layers_2, args.dropout, args.alpha, args.bns)
        self.res_fc = nn.Linear(nfeat, dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for gate in self.gates:
            nn.init.xavier_uniform_(gate.weight, gain=gain)
            nn.init.zeros_(gate.bias)
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_jk_ref.reset_parameters()
        self.lr_output.reset_parameters()
        if self.graph_process is not None:
            for layer in self.graph_process:
                layer.reset_parameters()
            for layer in self.ret_process:
                layer.reset_parameters()

    def forward(self, graph_features, retrieved_features, return_attention=False):
        graph_features = [self.input_drop(x) for x in graph_features]
        retrieved_features = [self.retrieval_drop(x) for x in retrieved_features]
        if self.pre_process:
            graph_inputs = [self.graph_process[i](graph_features[i]) for i in range(self.num_hops)]
            ret_inputs = [self.ret_process[i](retrieved_features[i]) for i in range(self.num_hops)]
        else:
            graph_inputs = graph_features
            ret_inputs = retrieved_features

        fused = []
        retrieval_gates = []
        for i in range(self.num_hops):
            gate = torch.sigmoid(self.gates[i](torch.cat([graph_inputs[i], ret_inputs[i]], dim=1)))
            fused_i = graph_inputs[i] + gate * ret_inputs[i]
            fused.append(fused_i)
            retrieval_gates.append(gate.mean(dim=1, keepdim=True))

        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(torch.cat(fused, dim=1))))
        scores = [self.act(self.lr_att(torch.cat([jk_ref, x], dim=1))) for x in fused]
        hop_weights = F.softmax(torch.cat(scores, dim=1), dim=1)
        hidden = fused[0] * self.att_drop(hop_weights[:, 0:1])
        for i in range(1, self.num_hops):
            hidden = hidden + fused[i] * self.att_drop(hop_weights[:, i:i + 1])
        if self.residual:
            hidden = self.dropout(self.prelu(hidden + self.res_fc(graph_features[0])))
        out = self.lr_output(hidden)
        if return_attention:
            return out, hop_weights, torch.cat(retrieval_gates, dim=1)
        return out


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
def build_retrieval_embedding(feats, mode):
    if mode == "hop0":
        emb = feats[0].float()
    else:
        emb = torch.stack([x.float() for x in feats], dim=0).mean(dim=0)
    return F.normalize(emb, p=2, dim=1)


@torch.no_grad()
def retrieve_with_faiss(emb, topk, args, logger):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is not installed") from exc
    logger.info("Building FAISS retrieval index: %s", args.faiss_index)
    emb_np = emb.cpu().numpy().astype("float32", copy=False)
    if args.faiss_index == "hnsw":
        index = faiss.IndexHNSWFlat(emb_np.shape[1], args.faiss_hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = args.faiss_ef_search
    else:
        index = faiss.IndexFlatIP(emb_np.shape[1])
    index.add(emb_np)
    sims, neigh = index.search(emb_np, topk + 1)
    neigh = torch.from_numpy(neigh).long()
    sims = torch.from_numpy(sims).float()
    return remove_self_neighbors(neigh, sims, topk)


@torch.no_grad()
def retrieve_with_torch(emb, topk, chunk_size, logger):
    logger.info("Running exact torch cosine retrieval in chunks")
    emb = emb.cpu().float()
    all_idx = []
    all_scores = []
    for start in range(0, emb.size(0), chunk_size):
        end = min(start + chunk_size, emb.size(0))
        sim = emb[start:end].matmul(emb.t())
        row = torch.arange(start, end)
        sim[torch.arange(end - start), row] = -float("inf")
        scores, idx = torch.topk(sim, k=topk, dim=1)
        all_idx.append(idx.cpu())
        all_scores.append(scores.cpu())
        logger.info("Retrieved rows %d:%d", start, end)
        del sim
        gc.collect()
    return torch.cat(all_idx, dim=0), torch.cat(all_scores, dim=0)


@torch.no_grad()
def remove_self_neighbors(neigh, sims, topk):
    rows = torch.arange(neigh.size(0)).view(-1, 1)
    sims = sims.masked_fill(neigh == rows, -float("inf"))
    clean_sims, pos = torch.topk(sims, k=topk, dim=1)
    clean_idx = torch.gather(neigh, 1, pos)
    return clean_idx, clean_sims


@torch.no_grad()
def compute_retrieval(feats, args, logger):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = "amazon_computer"
    cache_key = (
        f"{cache_name}_retrieval_h{args.num_hops}_k{args.retrieval_topk}_"
        f"{args.retrieval_embedding}_{args.cache_dtype}.pt"
    )
    cache_path = cache_dir / cache_key
    if args.cache_retrieval and cache_path.exists():
        logger.info("Loading cached retrieval tensors from %s", cache_path)
        payload = torch.load(cache_path, map_location="cpu")
        return payload["retrieved_features"], payload["neighbors"], payload["scores"]

    emb = build_retrieval_embedding(feats, args.retrieval_embedding)
    backend = args.retrieval_backend
    if backend in ("auto", "faiss"):
        try:
            neighbors, scores = retrieve_with_faiss(emb, args.retrieval_topk, args, logger)
            backend = "faiss"
        except RuntimeError:
            if args.retrieval_backend == "faiss":
                raise
            logger.info("FAISS unavailable; falling back to torch retrieval")
            neighbors, scores = retrieve_with_torch(emb, args.retrieval_topk, args.retrieval_chunk_size, logger)
            backend = "torch"
    else:
        neighbors, scores = retrieve_with_torch(emb, args.retrieval_topk, args.retrieval_chunk_size, logger)

    weights = F.softmax(scores / args.retrieval_temp, dim=1).float()
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    retrieved_features = []
    out_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    logger.info("Aggregating retrieved hop features with backend=%s", backend)
    for hop, feat in enumerate(feats):
        start = time.time()
        chunks = []
        for chunk_start in range(0, feat.size(0), args.aggregation_chunk_size):
            chunk_end = min(chunk_start + args.aggregation_chunk_size, feat.size(0))
            neigh_chunk = neighbors[chunk_start:chunk_end]
            weight_chunk = weights[chunk_start:chunk_end]
            agg = (feat[neigh_chunk].float() * weight_chunk.unsqueeze(-1)).sum(dim=1)
            chunks.append(agg.to(out_dtype).cpu())
        retrieved_features.append(torch.cat(chunks, dim=0))
        logger.info("Aggregated retrieved hop %d in %.2fs", hop, time.time() - start)
        gc.collect()

    if args.cache_retrieval:
        torch.save(
            {"retrieved_features": retrieved_features, "neighbors": neighbors, "scores": scores, "backend": backend},
            cache_path,
        )
        logger.info("Saved retrieval cache to %s", cache_path)
    return retrieved_features, neighbors, scores


def ogb_acc(evaluator, y_true, y_pred):
    return evaluator.eval({"y_true": y_true.view(-1, 1), "y_pred": y_pred.view(-1, 1)})["acc"]


def run_batches(indices, batch_size, shuffle):
    return torch.utils.data.DataLoader(indices.cpu(), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_epoch(model, feats, ret_feats, labels, train_idx, optimizer, evaluator, batch_size, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    y_true, y_pred = [], []
    for batch in run_batches(train_idx, batch_size, shuffle=True):
        graph_batch = [feat[batch].to(device) for feat in feats]
        ret_batch = [feat[batch].float().to(device) for feat in ret_feats]
        y = labels[batch].to(device)
        out = model(graph_batch, ret_batch)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.numel()
        total_examples += batch.numel()
        y_true.append(y.detach().cpu())
        y_pred.append(out.argmax(dim=-1).detach().cpu())
    train_acc = ogb_acc(evaluator, torch.cat(y_true), torch.cat(y_pred))
    return total_loss / max(total_examples, 1), train_acc


@torch.no_grad()
def evaluate(model, feats, ret_feats, labels, idx, evaluator, batch_size, device):
    model.eval()
    preds = []
    for batch in run_batches(idx, batch_size, shuffle=False):
        graph_batch = [feat[batch].to(device) for feat in feats]
        ret_batch = [feat[batch].float().to(device) for feat in ret_feats]
        preds.append(model(graph_batch, ret_batch).argmax(dim=-1).cpu())
    pred = torch.cat(preds)
    return ogb_acc(evaluator, labels[idx.cpu()], pred)


@torch.no_grad()
def predict_logits(model, feats, ret_feats, batch_size, device):
    model.eval()
    logits = []
    hop_weights = []
    ret_gates = []
    all_idx = torch.arange(feats[0].size(0))
    for batch in run_batches(all_idx, batch_size, shuffle=False):
        graph_batch = [feat[batch].to(device) for feat in feats]
        ret_batch = [feat[batch].float().to(device) for feat in ret_feats]
        out, hop_w, gate = model(graph_batch, ret_batch, return_attention=True)
        logits.append(out.cpu())
        hop_weights.append(hop_w.cpu())
        ret_gates.append(gate.cpu())
    return torch.cat(logits, dim=0), torch.cat(hop_weights, dim=0), torch.cat(ret_gates, dim=0)


def train_model(args, run_id, device):
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}_retrieval"
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger("retrieval_gamlp", out_dir)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))
    set_seed(args.seed + run_id)

    data, labels, split_idx, num_classes = load_products(args.dataset_root, logger, split_seed=args.seed + run_id)
    split_file = Path(args.split_file) if args.split_file else None
    if split_file and split_file.is_file():
        split_idx = load_split_idx_csv(split_file)
        logger.info("Fixed split from %s  train=%d valid=%d test=%d",
                    split_file, split_idx["train"].numel(), split_idx["valid"].numel(), split_idx["test"].numel())

    feats = precompute_features(data, args.num_hops, args.cache_dir, args.cache_features, logger)
    ret_feats, neighbors, scores = compute_retrieval(feats, args, logger)
    labels = labels.cpu()
    split_idx = {k: v.cpu() for k, v in split_idx.items()}
    evaluator = AccuracyEvaluator()

    model = RetrievalGAMLP(feats[0].size(1), args.hidden, num_classes, args.num_hops + 1, args).to(device)
    logger.info("Model params: %d", count_params(model))

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        result = {
            "train_acc": evaluate(model, feats, ret_feats, labels, split_idx["train"], evaluator, args.batch_size, device),
            "valid_acc": evaluate(model, feats, ret_feats, labels, split_idx["valid"], evaluator, args.batch_size, device),
            "test_acc": evaluate(model, feats, ret_feats, labels, split_idx["test"], evaluator, args.batch_size, device),
        }
        write_json(out_dir / "eval_results.json", result)
        logger.info("Eval-only results: %s", result)
        return result

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    stale = 0
    metrics_path = out_dir / "metrics.jsonl"
    ckpt_path = out_dir / "best_model.pt"
    for epoch in range(args.epochs):
        start = time.time()
        loss, train_acc = train_epoch(
            model, feats, ret_feats, labels, split_idx["train"], optimizer, evaluator, args.batch_size, device
        )
        val_acc = None
        if epoch % args.eval_every == 0:
            val_acc = evaluate(model, feats, ret_feats, labels, split_idx["valid"], evaluator, args.batch_size, device)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_test = evaluate(model, feats, ret_feats, labels, split_idx["test"], evaluator, args.batch_size, device)
                torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
                stale = 0
            else:
                stale += args.eval_every
        elapsed = time.time() - start
        payload = {
            "epoch": epoch, "loss": loss, "train_acc": train_acc, "val_acc": val_acc,
            "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch, "time_sec": elapsed,
        }
        append_jsonl(metrics_path, payload)
        logger.info(
            "epoch=%d loss=%.4f train=%.4f val=%s best_val=%.4f best_test=%.4f time=%.2fs",
            epoch, loss, train_acc, "None" if val_acc is None else f"{val_acc:.4f}",
            best_val, best_test, elapsed,
        )
        if stale >= args.patience:
            logger.info("Early stopping at epoch=%d", epoch)
            break

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    logits, hop_weights, ret_gates = predict_logits(model, feats, ret_feats, args.batch_size, device)
    torch.save(logits, out_dir / "logits.pt")
    np.save(out_dir / "logits.npy", logits.numpy())
    torch.save(hop_weights, out_dir / "hop_attention.pt")
    torch.save(ret_gates, out_dir / "retrieval_gates.pt")
    torch.save(neighbors, out_dir / "retrieval_neighbors.pt")
    torch.save(scores, out_dir / "retrieval_scores.pt")
    result = {
        "run": run_id, "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path), "output_dir": str(out_dir),
        "mean_hop_attention": hop_weights.mean(dim=0).tolist(),
        "mean_retrieval_gate": ret_gates.mean(dim=0).tolist(),
    }
    write_json(out_dir / "results.json", result)
    plot_path = plot_training_curves(metrics_path, out_dir, title="Retrieval-Guided GAMLP")
    if plot_path:
        logger.info("Saved training curves: %s", plot_path)
    logger.info("Final result: %s", result)
    return result


def main():
    args = parse_args()
    device = get_device(args.gpu)
    results = [train_model(args, run_id, device) for run_id in range(args.num_runs)]
    if len(results) > 1 and "best_test" in results[0]:
        vals = np.array([r["best_val"] for r in results], dtype=np.float64)
        tests = np.array([r["best_test"] for r in results], dtype=np.float64)
        print(json.dumps({
            "valid_mean": float(vals.mean()), "valid_std": float(vals.std()),
            "test_mean": float(tests.mean()), "test_std": float(tests.std()),
        }, indent=2))


if __name__ == "__main__":
    main()
