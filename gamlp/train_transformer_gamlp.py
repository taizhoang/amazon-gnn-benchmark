"""Train a lean multi-operator Transformer GAMLP model on Amazon Computers.

The model is intentionally plain-only:
multi-operator propagation -> multi-hop tokens -> Transformer -> attention
pooling -> residual gated classifier.  It omits RLU, label propagation,
teacher logits, and multi-stage distillation.
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GAMLP_original.train_gamlp_products import (
    AccuracyEvaluator,
    append_jsonl,
    build_mean_adj,
    evaluate,
    make_output_dir,
    predict_logits,
    run_batches,
    set_seed,
    train_epoch,
    write_json,
)
from load_dataset import load_products


def parse_args():
    parser = argparse.ArgumentParser(description="Lean multi-operator Transformer GAMLP")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/transformer_gamlp")
    parser.add_argument("--cache-dir", type=str, default="outputs/cache")
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-hops", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--token-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-multiplier", type=int, default=2)
    parser.add_argument("--classifier-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--input-drop", type=float, default=0.15)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--hop-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100)
    return parser.parse_args()


def setup_logger(out_dir):
    logger = logging.getLogger("lean_transformer_gamlp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@torch.no_grad()
def precompute_multi_operator_features(data, num_hops, cache_dir, cache_features, logger):
    """Create raw, mean-normalized, and symmetric-normalized hop tokens."""
    cache_name = getattr(data, "cache_name", "graph")
    cache_path = Path(cache_dir) / f"{cache_name}_multi_operator_hops_{num_hops}.pt"
    if cache_features and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if isinstance(cached, dict) and "features" in cached:
            logger.info("Loading cached multi-operator features from %s", cache_path)
            return cached["features"], cached["token_names"]
        logger.warning("Ignoring incompatible feature cache at %s", cache_path)

    adj_t, degree = build_mean_adj(data.edge_index, data.num_nodes)
    raw = data.x.float().cpu()
    mean_features, symmetric_features = [raw], []
    symmetric = raw
    inverse_sqrt_degree = degree.rsqrt()
    for hop in range(1, num_hops + 1):
        start = time.time()
        mean_features.append(adj_t.matmul(mean_features[-1]) / degree)
        symmetric = inverse_sqrt_degree * adj_t.matmul(inverse_sqrt_degree * symmetric)
        symmetric_features.append(symmetric)
        logger.info("Computed multi-operator hop %d in %.2fs", hop, time.time() - start)

    features = mean_features + symmetric_features
    token_names = ([f"mean_hop_{hop}" for hop in range(num_hops + 1)] +
                   [f"sym_hop_{hop}" for hop in range(1, num_hops + 1)])
    if cache_features:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "token_names": token_names}, cache_path)
        logger.info("Saved multi-operator feature cache to %s", cache_path)
    return features, token_names


class GatedResidualBlock(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.expand = nn.Linear(hidden, 2 * hidden)
        self.project = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, value = self.expand(self.norm(x)).chunk(2, dim=-1)
        return x + self.dropout(self.project(F.gelu(gate) * value))


class TransformerGAMLP(nn.Module):
    """Multi-hop token Transformer with attention pooling and gated classifier."""
    def __init__(self, in_feats, hidden, nclass, num_tokens, args):
        super().__init__()
        if hidden % args.num_heads:
            raise ValueError("--hidden must be divisible by --num-heads")
        if args.token_layers < 0:
            raise ValueError("--token-layers must be non-negative")
        self.num_tokens = num_tokens
        self.input_drop = nn.Dropout(args.input_drop)
        self.hop_drop_prob = args.hop_dropout
        self.token_projection = nn.Sequential(nn.Linear(in_feats, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.token_position = nn.Parameter(torch.empty(1, num_tokens, hidden))
        if args.token_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=args.num_heads, dim_feedforward=hidden * args.ffn_multiplier,
                dropout=args.attention_dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(layer, num_layers=args.token_layers, norm=nn.LayerNorm(hidden))
        else:
            self.token_encoder = nn.Identity()
        self.pool_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.classifier_input = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.classifier = nn.ModuleList(GatedResidualBlock(hidden, args.dropout) for _ in range(args.classifier_layers))
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, nclass)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token_position, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feature_list, label_emb=None):
        """Classify multi-hop feature tokens; label_emb is unused in plain mode."""
        if len(feature_list) != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} feature tokens, received {len(feature_list)}")
        tokens = torch.stack([self.input_drop(feature) for feature in feature_list], dim=1)
        tokens = self.token_projection(tokens) + self.token_position
        if self.training and self.hop_drop_prob:
            keep = torch.rand(tokens.shape[:2], device=tokens.device) >= self.hop_drop_prob
            keep[:, 0] = True
            tokens = tokens * keep.unsqueeze(-1)
        tokens = self.token_encoder(tokens)
        weights = F.softmax(self.pool_score(tokens).squeeze(-1), dim=1)
        fused = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        hidden = self.classifier_input(fused + tokens[:, 0])
        for block in self.classifier:
            hidden = block(hidden)
        return self.output(self.output_norm(hidden))


def train_model(args, model, features, labels, split_idx, evaluator, out_dir, logger, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_test, best_epoch, stale = -1.0, -1.0, -1, 0
    checkpoint_path = out_dir / "best_model.pt"
    metrics_path = out_dir / "metrics.jsonl"
    for epoch in range(args.epochs):
        start = time.time()
        loss, train_acc = train_epoch(model, features, labels, None, split_idx["train"], optimizer, evaluator, args.batch_size, device)
        val_acc = None
        if epoch % args.eval_every == 0:
            val_acc = evaluate(model, features, labels, None, split_idx["valid"], evaluator, args.batch_size, device)
            if val_acc > best_val:
                best_val, best_epoch, stale = val_acc, epoch, 0
                best_test = evaluate(model, features, labels, None, split_idx["test"], evaluator, args.batch_size, device)
                torch.save({"model_state": model.state_dict(), "args": vars(args)}, checkpoint_path)
            else:
                stale += args.eval_every
        payload = {"epoch": epoch, "loss": loss, "train_acc": train_acc, "val_acc": val_acc,
                   "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch,
                   "time_sec": time.time() - start}
        append_jsonl(metrics_path, payload)
        logger.info("epoch=%d loss=%.4f train=%.4f val=%s best_val=%.4f best_test=%.4f time=%.2fs",
                    epoch, loss, train_acc, "None" if val_acc is None else f"{val_acc:.4f}",
                    best_val, best_test, payload["time_sec"])
        if stale >= args.patience:
            logger.info("Early stopping at epoch=%d", epoch)
            break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return best_val, best_test, best_epoch, checkpoint_path


def run_once(args, run_id, device):
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_id}_transformer"
    out_dir = make_output_dir(args.output_dir, run_name)
    logger = setup_logger(out_dir)
    set_seed(args.seed + run_id)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))
    data, labels, split_idx, num_classes = load_products(args.dataset_root, logger, split_seed=args.seed + run_id)
    features, token_names = precompute_multi_operator_features(data, args.num_hops, args.cache_dir, args.cache_features, logger)
    labels = labels.cpu()
    split_idx = {name: index.cpu() for name, index in split_idx.items()}
    model = TransformerGAMLP(features[0].size(1), args.hidden, num_classes, len(features), args).to(device)
    logger.info("Using tokens: %s", token_names)
    logger.info("Model parameters: %d", sum(parameter.numel() for parameter in model.parameters()))
    best_val, best_test, best_epoch, checkpoint_path = train_model(
        args, model, features, labels, split_idx, AccuracyEvaluator(), out_dir, logger, device
    )
    logits = predict_logits(model, features, None, args.batch_size, device)
    torch.save(logits, out_dir / "logits.pt")
    np.save(out_dir / "logits.npy", logits.numpy())
    result = {"run": run_id, "method": "lean_multi_operator_transformer_gamlp", "feature_tokens": token_names,
              "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch,
              "checkpoint": str(checkpoint_path), "output_dir": str(out_dir)}
    write_json(out_dir / "results.json", result)
    logger.info("Final result: %s", result)
    gc.collect()
    return result


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    results = [run_once(args, run_id, device) for run_id in range(args.num_runs)]
    if len(results) > 1:
        values = np.array([[result["best_val"], result["best_test"]] for result in results])
        print(json.dumps({"valid_mean": float(values[:, 0].mean()), "valid_std": float(values[:, 0].std()),
                          "test_mean": float(values[:, 1].mean()), "test_std": float(values[:, 1].std())}, indent=2))


if __name__ == "__main__":
    main()
