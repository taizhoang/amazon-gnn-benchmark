# amazon-gnn-benchmark

A benchmark of GNN architectures on **Amazon Computers** (co-purchase graph,
~13.7K products, 767-dim bag-of-words features, 10 categories), consolidated
from three earlier experiment repos into one shared codebase, common dataset
loader, and fixed train/valid/test split so every model is compared fairly.

Every model is implemented in plain PyTorch (`scatter_add_` / `scatter_reduce`
in [common/utils.py](common/utils.py)) — no `torch-geometric.nn`,
`torch-scatter`, `torch-sparse`, or DGL. Those compiled extensions must match
the installed torch build exactly, which breaks easily across environments;
avoiding them keeps every script here runnable with nothing beyond `torch`
itself and keeps peak memory at O(E) for the full graph.

## Model families

| Folder | Script | Model |
|---|---|---|
| [graphsage/](graphsage/) | `train_graphsage.py` | GraphSAGE (mean aggregator), full-batch or mini-batch neighbour sampling |
| [graphsage/](graphsage/) | `train_improved_graphsage.py` | + multi-aggregator (mean/max/std), Jumping Knowledge, degree-based importance sampling |
| [sagat/](sagat/) | `train_sagat.py` | GAT baseline / structural-feature-concat ablation / Structure-Aware GAT (structural bias injected into attention) |
| [gamlp/](gamlp/) | `train_gamlp.py` | GAMLP (decoupled propagation + hop-attention MLP), plain or RLU (label propagation + self-training) mode |
| [gamlp/](gamlp/) | `train_transformer_gamlp.py` | Multi-operator (mean + symmetric normalisation) hop tokens fed through a Transformer encoder with attention pooling and a gated-residual classifier |
| [lightgcn/](lightgcn/) | `train_lightgcn_grand.py` | LightGCN layer-combination propagation trained with GRAND (DropNode + consistency regularisation), plus an optional class-Compatibility-Propagation (CoP) channel |

Each script is self-contained and runnable directly (`python graphsage/train_graphsage.py`),
writing to its own `outputs/<model>/<timestamp>_run.../` directory:
`train.log`, `metrics.jsonl` (per epoch), `results.json` (final, including
`num_params`), `curves.png` (loss/accuracy plot), and a `best.pt` /
`best_model.pt` checkpoint.

## Setup

```bash
pip install -r requirements.txt
```

The dataset (`amazon_co_buy_computer.npz`) is auto-downloaded into `data/`
via `torch_geometric.datasets.Amazon` the first time any script runs, if
`torch-geometric` is installed (see the commented-out line in
`requirements.txt`). Otherwise, place the file manually under
`data/amazon_co_buy_computer/amazon_co_buy_computer.npz`.

`split_idx.csv` at the repo root is a fixed stratified 60/20/20
train/valid/test split (produced once, saved to CSV) that every script
loads by default via `--split-file split_idx.csv`, so all models in the
comparison tables below are evaluated on the exact same node sets. Pass
`--split-file ""` to instead regenerate a fresh split from `--seed`.

## Running the benchmark

```bash
# GraphSAGE
python graphsage/train_graphsage.py
python graphsage/train_graphsage.py --sampling --batch-size 512 --fanout 10 10   # mini-batch mode

# Improved GraphSAGE (multi-aggregator + JK + importance sampling, all on by default)
python graphsage/train_improved_graphsage.py
python graphsage/train_improved_graphsage.py --no-multi-aggr --jk-mode none      # ablate back to plain GraphSAGE at the same width/depth

# SA-GAT: baseline GAT, structural-concat ablation, and structure-aware attention (ours)
python sagat/train_sagat.py --variant gat
python sagat/train_sagat.py --variant gat-concat
python sagat/train_sagat.py --variant sagat          # also runs the attention-vs-structure correlation analysis

# GAMLP baseline
python gamlp/train_gamlp.py --mode plain --cache-features
python gamlp/train_gamlp.py --mode rlu --cache-features

# Transformer GAMLP
python gamlp/train_transformer_gamlp.py --cache-features

# LightGCN-GRAND (CoP channel on by default; disable with --no-use-H)
python lightgcn/train_lightgcn_grand.py
python lightgcn/train_lightgcn_grand.py --low-label-split   # reproduce GRAND's original 20/30-per-class regime (not comparable to the rows above)
```

Every script exposes `--hidden`, `--dropout`, `--lr`, `--epochs`,
`--patience`, `--seed`, `--num-runs`, `--gpu` (`--gpu -1` forces CPU), plus
model-specific flags — run any script with `--help` for the full list.

## Notebooks

| Notebook | Purpose |
|---|---|
| [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) | Dataset exploration — split sizes, label distribution, feature stats, degree distribution, edge homophily. Loads through `common/data.py`, so the numbers match what every training script actually sees. |
| [notebooks/02_train_all_standalone.ipynb](notebooks/02_train_all_standalone.ipynb) | Trains every model **standalone** — every utility, the dataset loader, and all 6 model implementations are written directly in the notebook's own cells, no `import` from `common/`/`graphsage/`/`sagat/`/`gamlp/`/`lightgcn/`. Copy this one file (+ `split_idx.csv`) and it runs on its own, e.g. on Kaggle (a commented `%pip install` cell at the top covers a fresh environment). Defaults to a fast `SMOKE_TEST` pass over all 6 models; flip one flag for full training. |

▶ **Run `02_train_all_standalone.ipynb` on Kaggle:** https://www.kaggle.com/code/phanchanchan/amazon-gnn-benchmark

Both notebooks write their figures/tables/checkpoints under `outputs/` (gitignored), same as the training scripts.

## Provenance

Consolidated from three prior repos, each covering a different part of this
benchmark:

- **GraphML_GAMLP_v2** — `common/`, `graphsage/train_graphsage.py`,
  `graphsage/train_improved_graphsage.py` (the shared scatter-based utilities
  and dataset loader, and the GraphSAGE family, ported near-verbatim).
- **GRAPH-SAGAT** — `sagat/train_sagat.py` (Structure-Aware GAT concept —
  degree/PageRank/clustering/betweenness injected into GAT attention —
  reimplemented from scratch in pure PyTorch; the original prototype used
  DGL and computed the same idea with `edge_softmax`).
- **GraphML_subject** — `gamlp/train_gamlp.py`, `gamlp/train_transformer_gamlp.py`
  (GAMLP baseline and the Transformer-over-hop-tokens variant, ported with the
  torch-sparse `SparseTensor` hop-propagation replaced by `common.utils.propagate_mean`
  / `weighted_propagate`, plain `scatter_add`-based equivalents — exact, not
  approximate, since this dataset's adjacency is symmetric; see
  `propagate_mean`'s docstring). An earlier retrieval-guided GAMLP variant was
  ported first but was replaced upstream in GraphML_subject by the Transformer
  variant, so this repo now tracks that instead.
- `lightGCN-GRAND.py` (repo root) — a TensorFlow prototype, ported to pure
  PyTorch as `lightgcn/train_lightgcn_grand.py` using the same
  `common/utils.py` scatter ops as everything else (new: `gcn_norm_edge_weight`
  + `weighted_propagate`), and switched from its original low-label-rate
  split to the benchmark's shared `split_idx.csv` by default (see
  `--low-label-split` to opt back into the original regime).
