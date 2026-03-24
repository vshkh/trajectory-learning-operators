"""
baseline_comparison.py
======================
Compares three standard baselines against the ArithmeticWorldModel (v2.1)
on the same trajectory prediction task.

Baselines
---------
  MLP        — flat MLP, context window → all 15 prediction steps at once
  LSTM       — recurrent, processes context step-by-step, unrolls for predictions
  Transformer — encoder-only with positional encoding, attends over context,
                linear head predicts all 15 steps at once

Fairness constraints
--------------------
  - Identical train/test split (same SEED, same PerOpNormaliser)
  - Identical input (CTX_LEN=5 normalised context) and target (PRED_LEN=15)
  - Parameter counts matched to ±2× of ArithmeticWorldModel (~22K params)
  - Identical optimiser (Adam, cosine annealing, grad clip 1.0)
  - Identical epochs (250) and batch size (64)
  - All baselines predict all 15 steps in one forward pass (same eval protocol
    as ArithmeticWorldModel, which unrolls autoregressively but is evaluated
    on the 15-step MSE — baselines use direct multi-step output for fairness;
    we also test LSTM in autoregressive mode separately)

Evaluation dimensions
---------------------
  1. In-distribution MSE   — test set from same x0 ∈ [1,10]
  2. OOD x0 MSE            — x0 ∈ {20, 100, 500}
  3. Temporal extrapolation — predict 60 steps (4× training horizon)
  4. Per-operation breakdown — which operation class does each model struggle on?
  5. Parameter count        — reported for fairness audit

Requirements: pip install torch numpy matplotlib pandas scikit-learn
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ── add parent dir so we can import shared utilities from main file ──────────
# If running from same directory as poc_arithmetic_dynamics.py, this is fine.
# Otherwise adjust the path below.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poc_arithmetic_dynamics import (
    TRAIN_OPS, HARD_OPS, CLIP_VAL,
    CTX_LEN, PRED_LEN, TRAJ_LEN, N_PER_OP, EPOCHS,
    generate_trajectory, make_dataset,
    PerOpNormaliser, split_tensors,
    ArithmeticWorldModel,          # our model
)

SEED    = 42
OUT_DIR = "baseline_results"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

CODE_DIM   = 12
LAMBDA_ENT = 0.01


def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved → {path}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════
# BASELINE ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════

class MLPBaseline(nn.Module):
    """
    Flat MLP: context window (5 values) → all PRED_LEN steps at once.

    No notion of dynamics, no recurrence, no attention.
    Represents the simplest possible learned function approximator.
    Target parameter count: ~20K.

    Design: 5 → 256 → 256 → 128 → PRED_LEN
    Params: 5*256 + 256*256 + 256*128 + 128*15 ≈ 100K  (oversized intentionally
            to give MLP the best possible chance — if it still fails, the point
            about inductive bias is stronger)
    """
    def __init__(self, ctx_len: int = CTX_LEN, pred_len: int = PRED_LEN,
                 hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_len, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, pred_len),
        )

    def forward(self, context: torch.Tensor, n_steps: int) -> torch.Tensor:
        # n_steps ignored — always outputs PRED_LEN steps
        # This is intentional: MLP has no mechanism for variable-length rollout
        return self.net(context)


class LSTMBaseline(nn.Module):
    """
    LSTM: processes context step-by-step, then unrolls to predict future steps.

    Two modes:
      direct   — after encoding context, a linear layer predicts all pred_len
                 steps at once from the final hidden state. Fast, no error
                 accumulation. Most favourable to LSTM.
      autoregressive — after encoding context, feeds own predictions back in
                       one step at a time. Error-accumulating, same as our model.
                       We report both; autoregressive is the fair structural
                       comparison.

    Hidden size 64, 1 layer → ~22K params (matched to our model).
    """
    def __init__(self, hidden: int = 64, pred_len: int = PRED_LEN,
                 mode: str = "direct"):
        super().__init__()
        assert mode in ("direct", "autoregressive")
        self.mode     = mode
        self.hidden   = hidden
        self.pred_len = pred_len
        # Input is a single scalar at each step
        self.lstm     = nn.LSTM(input_size=1, hidden_size=hidden,
                                num_layers=1, batch_first=True)
        if mode == "direct":
            self.head = nn.Linear(hidden, pred_len)
        else:
            # One-step decoder
            self.decoder_lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                                        num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, 1)

    def forward(self, context: torch.Tensor, n_steps: int) -> torch.Tensor:
        B = context.shape[0]
        # Encode context: (B, CTX_LEN) → (B, CTX_LEN, 1)
        x = context.unsqueeze(-1)
        _, (h, c) = self.lstm(x)   # h: (1, B, hidden)

        if self.mode == "direct":
            out = self.head(h.squeeze(0))    # (B, pred_len)
            return out

        else:  # autoregressive
            preds  = []
            h_dec  = h
            c_dec  = c
            # Seed with last context value
            z = context[:, -1:].unsqueeze(-1)   # (B, 1, 1)
            for _ in range(n_steps):
                out_dec, (h_dec, c_dec) = self.decoder_lstm(z, (h_dec, c_dec))
                z_next = self.head(out_dec[:, -1, :])   # (B, 1)
                preds.append(z_next)
                z = z_next.unsqueeze(1)
            return torch.cat(preds, dim=1)   # (B, n_steps)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerBaseline(nn.Module):
    """
    Encoder-only Transformer.

    Each of the 5 context values is a token embedded into d_model dimensions.
    Self-attention attends over all 5 tokens. The CLS-style final representation
    (mean pool over all tokens) is projected to all pred_len outputs at once.

    This is the most favourable Transformer formulation for this task — it sees
    the full context at once (no causal mask needed for the context window) and
    predicts all future steps simultaneously without autoregressive error.

    d_model=32, nhead=4, 2 encoder layers → ~18K params (matched to our model).
    """
    def __init__(self, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 64,
                 ctx_len: int = CTX_LEN, pred_len: int = PRED_LEN):
        super().__init__()
        self.d_model  = d_model
        self.pred_len = pred_len
        # Project scalar input to d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=ctx_len + 1)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, pred_len)

    def forward(self, context: torch.Tensor, n_steps: int) -> torch.Tensor:
        # context: (B, CTX_LEN)
        x = context.unsqueeze(-1)              # (B, CTX_LEN, 1)
        x = self.input_proj(x)                 # (B, CTX_LEN, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)                    # (B, CTX_LEN, d_model)
        pooled = x.mean(dim=1)                 # (B, d_model) — mean pool
        return self.head(pooled)               # (B, pred_len)


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train_baseline(
    model: nn.Module,
    tr_ctx: torch.Tensor,
    tr_tgt: torch.Tensor,
    te_ctx: torch.Tensor,
    te_tgt: torch.Tensor,
    n_pred: int,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
    bs: int = 64,
    label: str = "",
    verbose: bool = True,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Identical training protocol to ArithmeticWorldModel:
      Adam + cosine annealing + grad clip 1.0 + NaN guard.
    No entropy loss (baselines have no gate to regularise).
    """
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n_train = len(tr_ctx)
    tr_hist, te_hist = [], []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0.0, 0

        for i in range(0, n_train, bs):
            idx   = perm[i: i + bs]
            ctx_b = tr_ctx[idx]
            tgt_b = tr_tgt[idx]
            if not (torch.isfinite(ctx_b).all() and torch.isfinite(tgt_b).all()):
                continue
            pred = model(ctx_b, n_pred)
            # Trim/pad if model outputs different length (shouldn't happen but safe)
            pred = pred[:, :n_pred]
            loss = loss_fn(pred, tgt_b[:, :n_pred])
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb      += 1

        sched.step()
        model.eval()
        with torch.no_grad():
            te_pred = model(te_ctx, n_pred)[:, :n_pred]
            if torch.isfinite(te_pred).all():
                te_loss = loss_fn(te_pred, te_tgt[:, :n_pred]).item()
            else:
                te_loss = float("nan")

        tr_hist.append(ep_loss / max(nb, 1))
        te_hist.append(te_loss)

        if verbose and (ep + 1) % 50 == 0:
            tag = f"[{label}] " if label else ""
            print(f"    {tag}Epoch {ep+1:4d} | "
                  f"Train {tr_hist[-1]:.6f} | Test {te_loss:.6f}")

    return model, tr_hist, te_hist


def train_our_model(
    tr_ctx, tr_tgt, te_ctx, te_tgt, n_pred,
    epochs=EPOCHS, lr=1e-3, bs=64, label="our_model",
) -> Tuple[ArithmeticWorldModel, List[float], List[float]]:
    """Train ArithmeticWorldModel with entropy regularisation."""
    model   = ArithmeticWorldModel(CTX_LEN, CODE_DIM)
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n_train = len(tr_ctx)
    tr_hist, te_hist = [], []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss, nb = 0.0, 0
        for i in range(0, n_train, bs):
            idx   = perm[i: i + bs]
            ctx_b = tr_ctx[idx]
            tgt_b = tr_tgt[idx]
            if not (torch.isfinite(ctx_b).all() and torch.isfinite(tgt_b).all()):
                continue
            pred, _, gate_probs = model(ctx_b, n_pred, return_extras=True)
            pred_loss  = loss_fn(pred, tgt_b)
            gate_ent   = -torch.mean(
                torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=-1))
            loss = pred_loss - LAMBDA_ENT * gate_ent
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb      += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            te_pred = model(te_ctx, n_pred)
            te_loss = loss_fn(te_pred, te_tgt).item() \
                if torch.isfinite(te_pred).all() else float("nan")
        tr_hist.append(ep_loss / max(nb, 1))
        te_hist.append(te_loss)

        if (ep + 1) % 50 == 0:
            print(f"    [{label}] Epoch {ep+1:4d} | "
                  f"Train {tr_hist[-1]:.6f} | Test {te_loss:.6f}")

    return model, tr_hist, te_hist


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def eval_per_operation(
    model: nn.Module,
    norm: PerOpNormaliser,
    ops: OrderedDict,
    n_pred: int,
    n_samples: int = 80,
    is_our_model: bool = False,
) -> pd.DataFrame:
    """
    For each operation, generate n_samples test trajectories from x0 ∈ [1,10]
    and compute mean MAE in normalised space.
    Returns DataFrame: operation | mae_normalised
    """
    model.eval()
    rows = []
    for op_idx, (name, fn) in enumerate(ops.items()):
        mu, sigma = norm.get(op_idx)
        batch_ctx, batch_tgt = [], []
        for _ in range(n_samples):
            x0 = np.random.uniform(1.0, 10.0)
            t  = generate_trajectory(fn, x0, TRAJ_LEN)
            tn = (t - mu) / sigma
            if np.all(np.isfinite(tn)):
                batch_ctx.append(tn[:CTX_LEN])
                batch_tgt.append(tn[CTX_LEN:])
        if not batch_ctx:
            continue
        ctx_t = torch.from_numpy(np.array(batch_ctx, dtype=np.float32))
        tgt_t = torch.from_numpy(np.array(batch_tgt, dtype=np.float32))
        with torch.no_grad():
            pred  = model(ctx_t, n_pred)
            pred  = pred[:, :n_pred]
            mae   = torch.abs(pred - tgt_t[:, :n_pred]).mean().item()
        rows.append({"operation": name, "mae_normalised": mae})
    return pd.DataFrame(rows)


def eval_ood(
    model: nn.Module,
    norm: PerOpNormaliser,
    ops: OrderedDict,
    n_pred: int,
    ood_x0s: List[float],
    n_samples: int = 40,
) -> pd.DataFrame:
    """MAE per operation per OOD x0."""
    model.eval()
    rows = []
    for op_idx, (name, fn) in enumerate(ops.items()):
        mu, sigma = norm.get(op_idx)
        row = {"operation": name}
        for x0 in ood_x0s:
            maes = []
            for _ in range(n_samples):
                t  = generate_trajectory(fn, x0, TRAJ_LEN)
                tn = (t - mu) / sigma
                if not np.all(np.isfinite(tn)):
                    continue
                ctx_t = torch.from_numpy(tn[:CTX_LEN]).unsqueeze(0).float()
                tgt   = tn[CTX_LEN:]
                with torch.no_grad():
                    pred = model(ctx_t, n_pred).squeeze(0)[:n_pred].numpy()
                maes.append(float(np.nanmean(np.abs(pred - tgt[:n_pred]))))
            row[f"x0_{int(x0)}"] = float(np.mean(maes)) if maes else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def eval_temporal_extrap(
    model: nn.Module,
    norm: PerOpNormaliser,
    ops: OrderedDict,
    horizon: int = 4 * PRED_LEN,   # 60 steps
    x0: float = 5.0,
) -> pd.DataFrame:
    """
    Predict horizon steps (4× training length).
    Report MAE in three bands: 1-15 (training), 16-30, 31-60.
    """
    model.eval()
    rows = []
    for op_idx, (name, fn) in enumerate(ops.items()):
        mu, sigma = norm.get(op_idx)
        true_long = generate_trajectory(fn, x0, CTX_LEN + horizon)
        true_n    = (true_long - mu) / sigma
        if not np.all(np.isfinite(true_n)):
            continue
        ctx_t = torch.from_numpy(true_n[:CTX_LEN]).unsqueeze(0).float()
        with torch.no_grad():
            pred_n = model(ctx_t, horizon).squeeze(0)[:horizon].numpy()
        fut = true_n[CTX_LEN:]
        row = {"operation": name}
        for seg, lo, hi in [
            ("steps_01_15",  0,       PRED_LEN),
            ("steps_16_30",  PRED_LEN,   2*PRED_LEN),
            ("steps_31_60",  2*PRED_LEN, horizon),
        ]:
            if hi <= len(pred_n) and hi <= len(fut):
                row[seg] = float(np.nanmean(np.abs(pred_n[lo:hi] - fut[lo:hi])))
            else:
                row[seg] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

COLORS = {
    "our_model":       "#1f77b4",   # blue
    "mlp":             "#d62728",   # red
    "lstm_direct":     "#ff7f0e",   # orange
    "lstm_autoreg":    "#e377c2",   # pink
    "transformer":     "#2ca02c",   # green
}

MODEL_LABELS = {
    "our_model":       "Ours (Hierarchical)",
    "mlp":             "MLP baseline",
    "lstm_direct":     "LSTM (direct)",
    "lstm_autoreg":    "LSTM (autoregressive)",
    "transformer":     "Transformer baseline",
}


def plot_training_curves(hists: Dict[str, List[float]], title: str, fname: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, hist in hists.items():
        finite = [v for v in hist if np.isfinite(v)]
        if finite:
            ax.semilogy(finite, label=MODEL_LABELS.get(key, key),
                        color=COLORS.get(key, None), alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE (log)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fname)


def plot_per_op_comparison(dfs: Dict[str, pd.DataFrame], fname: str):
    """Grouped bar chart: MAE per operation per model."""
    ops    = dfs[list(dfs.keys())[0]]["operation"].tolist()
    models = list(dfs.keys())
    n_ops  = len(ops)
    n_mod  = len(models)
    x      = np.arange(n_ops)
    width  = 0.8 / n_mod

    fig, ax = plt.subplots(figsize=(max(12, n_ops * 1.2), 5))
    for i, key in enumerate(models):
        vals = dfs[key].set_index("operation").reindex(ops)["mae_normalised"].values
        ax.bar(x + i * width - (n_mod - 1) * width / 2,
               np.log1p(vals.astype(float)),
               width * 0.9,
               label=MODEL_LABELS.get(key, key),
               color=COLORS.get(key, None),
               alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("log(1 + MAE) — normalised space")
    ax.set_title("Per-operation MAE: all models\n"
                 "(lower is better; log scale to handle mul_* outliers)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    savefig(fname)


def plot_ood_comparison(dfs: Dict[str, pd.DataFrame],
                        ops_subset: List[str], fname: str):
    """Line plot: MAE vs OOD x0 for selected operations."""
    ood_cols = [c for c in dfs[list(dfs.keys())[0]].columns if c.startswith("x0_")]
    x0_vals  = [int(c.split("_")[1]) for c in ood_cols]
    n_ops    = len(ops_subset)
    fig, axes = plt.subplots(1, n_ops, figsize=(n_ops * 4, 4), sharey=False)
    if n_ops == 1:
        axes = [axes]
    for ax, op in zip(axes, ops_subset):
        for key, df in dfs.items():
            row = df[df["operation"] == op]
            if row.empty:
                continue
            vals = [float(row[c].values[0]) for c in ood_cols]
            ax.semilogy(x0_vals, vals,
                        "o-", label=MODEL_LABELS.get(key, key),
                        color=COLORS.get(key, None), markersize=5)
        ax.set_title(op, fontsize=10)
        ax.set_xlabel("OOD x₀")
        ax.set_ylabel("MAE (normalised, log)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("OOD x₀ extrapolation: MAE vs initial condition\n"
                 "(trained on x₀ ∈ [1,10])", fontsize=11)
    plt.tight_layout()
    savefig(fname)


def plot_temporal_comparison(dfs: Dict[str, pd.DataFrame],
                              ops_subset: List[str], fname: str):
    """Bar chart: MAE per temporal band per model for selected ops."""
    bands  = ["steps_01_15", "steps_16_30", "steps_31_60"]
    n_ops  = len(ops_subset)
    models = list(dfs.keys())
    n_mod  = len(models)
    fig, axes = plt.subplots(1, n_ops, figsize=(n_ops * 5, 4))
    if n_ops == 1:
        axes = [axes]
    for ax, op in zip(axes, ops_subset):
        x     = np.arange(len(bands))
        width = 0.8 / n_mod
        for i, key in enumerate(models):
            df  = dfs[key]
            row = df[df["operation"] == op]
            if row.empty:
                continue
            vals = [float(row[b].values[0]) if b in row.columns else 0.0
                    for b in bands]
            ax.bar(x + i * width - (n_mod - 1) * width / 2,
                   np.log1p(vals),
                   width * 0.9,
                   label=MODEL_LABELS.get(key, key),
                   color=COLORS.get(key, None), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["1–15\n(train)", "16–30\n(+1×)", "31–60\n(+3×)"],
                           fontsize=9)
        ax.set_title(op, fontsize=10)
        ax.set_ylabel("log(1+MAE)")
        if i == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Temporal extrapolation: MAE by horizon band", fontsize=11)
    plt.tight_layout()
    savefig(fname)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  BASELINE COMPARISON")
    print("=" * 70)

    # ── Data ────────────────────────────────────────────────────────────
    print("\n  Generating data...")
    trajs, labels, op_names = make_dataset(TRAIN_OPS, N_PER_OP, TRAJ_LEN)
    norm    = PerOpNormaliser().fit(trajs, labels)
    trajs_n = norm.transform(trajs, labels)

    (tr_ctx, tr_tgt, tr_lbl,
     te_ctx, te_tgt, te_lbl) = split_tensors(trajs_n, labels, CTX_LEN)

    print(f"  Train {len(tr_ctx)} | Test {len(te_ctx)} | Ops {len(op_names)}")

    # ── Instantiate models ───────────────────────────────────────────────
    mlp          = MLPBaseline(CTX_LEN, PRED_LEN, hidden=128)
    lstm_direct  = LSTMBaseline(hidden=64, pred_len=PRED_LEN, mode="direct")
    lstm_autoreg = LSTMBaseline(hidden=64, pred_len=PRED_LEN, mode="autoregressive")
    transformer  = TransformerBaseline(d_model=32, nhead=4, num_layers=2,
                                       dim_ff=64, ctx_len=CTX_LEN, pred_len=PRED_LEN)
    our_dummy    = ArithmeticWorldModel(CTX_LEN, CODE_DIM)  # just for param count

    print("\n  Parameter counts:")
    param_table = {
        "our_model":    count_params(our_dummy),
        "mlp":          count_params(mlp),
        "lstm_direct":  count_params(lstm_direct),
        "lstm_autoreg": count_params(lstm_autoreg),
        "transformer":  count_params(transformer),
    }
    for k, v in param_table.items():
        print(f"    {MODEL_LABELS[k]:30s}: {v:,}")
    del our_dummy

    pd.DataFrame(
        [{"model": MODEL_LABELS[k], "n_params": v}
         for k, v in param_table.items()]
    ).to_csv(os.path.join(OUT_DIR, "table_param_counts.csv"), index=False)

    # ── Training ─────────────────────────────────────────────────────────
    te_hists = {}

    print("\n" + "="*70)
    print("  TRAINING: Our Model")
    print("="*70)
    our_model, _, te_h = train_our_model(
        tr_ctx, tr_tgt, te_ctx, te_tgt, PRED_LEN, label="our_model")
    te_hists["our_model"] = te_h

    print("\n" + "="*70)
    print("  TRAINING: MLP Baseline")
    print("="*70)
    mlp, _, te_h = train_baseline(
        mlp, tr_ctx, tr_tgt, te_ctx, te_tgt, PRED_LEN, label="mlp")
    te_hists["mlp"] = te_h

    print("\n" + "="*70)
    print("  TRAINING: LSTM Direct")
    print("="*70)
    lstm_direct, _, te_h = train_baseline(
        lstm_direct, tr_ctx, tr_tgt, te_ctx, te_tgt, PRED_LEN, label="lstm_direct")
    te_hists["lstm_direct"] = te_h

    print("\n" + "="*70)
    print("  TRAINING: LSTM Autoregressive")
    print("="*70)
    lstm_autoreg, _, te_h = train_baseline(
        lstm_autoreg, tr_ctx, tr_tgt, te_ctx, te_tgt, PRED_LEN,
        label="lstm_autoreg")
    te_hists["lstm_autoreg"] = te_h

    print("\n" + "="*70)
    print("  TRAINING: Transformer")
    print("="*70)
    transformer, _, te_h = train_baseline(
        transformer, tr_ctx, tr_tgt, te_ctx, te_tgt, PRED_LEN, label="transformer")
    te_hists["transformer"] = te_h

    # ── Training curves ──────────────────────────────────────────────────
    plot_training_curves(
        te_hists,
        "Test MSE during training — all models\n"
        "(same data, same optimiser, same epochs)",
        "fig_training_curves.png",
    )

    # ── In-distribution evaluation ───────────────────────────────────────
    print("\n" + "="*70)
    print("  EVALUATION: In-distribution per-operation MAE")
    print("="*70)

    models_dict = {
        "our_model":    our_model,
        "mlp":          mlp,
        "lstm_direct":  lstm_direct,
        "lstm_autoreg": lstm_autoreg,
        "transformer":  transformer,
    }

    per_op_dfs = {}
    for key, model in models_dict.items():
        df = eval_per_operation(model, norm, TRAIN_OPS, PRED_LEN,
                                is_our_model=(key == "our_model"))
        per_op_dfs[key] = df
        df.to_csv(os.path.join(OUT_DIR, f"table_per_op_{key}.csv"), index=False)
        print(f"\n  {MODEL_LABELS[key]}:")
        print(df.to_string(index=False))

    plot_per_op_comparison(per_op_dfs, "fig_per_op_comparison.png")

    # Combined summary table
    summary_rows = []
    for op in op_names:
        row = {"operation": op}
        for key, df in per_op_dfs.items():
            match = df[df["operation"] == op]
            row[MODEL_LABELS[key]] = (
                round(float(match["mae_normalised"].values[0]), 6)
                if not match.empty else float("nan")
            )
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(
        os.path.join(OUT_DIR, "table_summary_per_op.csv"), index=False)
    print("\n  SUMMARY (MAE normalised, in-distribution):")
    print(df_summary.to_string(index=False))

    # ── OOD x0 evaluation ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  EVALUATION: OOD x0 extrapolation")
    print("="*70)

    ood_x0s = [20.0, 100.0, 500.0]
    ood_dfs  = {}
    for key, model in models_dict.items():
        df = eval_ood(model, norm, TRAIN_OPS, PRED_LEN, ood_x0s)
        ood_dfs[key] = df
        df.to_csv(os.path.join(OUT_DIR, f"table_ood_{key}.csv"), index=False)

    # Plot for representative ops: one additive, one contracting, one multiplicative
    ood_showcase = [n for n in ["add_3", "div_2", "mul_2"] if n in op_names]
    plot_ood_comparison(ood_dfs, ood_showcase, "fig_ood_comparison.png")

    # ── Temporal extrapolation ───────────────────────────────────────────
    print("\n" + "="*70)
    print("  EVALUATION: Temporal extrapolation (60 steps)")
    print("="*70)

    temp_dfs = {}
    for key, model in models_dict.items():
        df = eval_temporal_extrap(model, norm, TRAIN_OPS, horizon=4*PRED_LEN)
        temp_dfs[key] = df
        df.to_csv(os.path.join(OUT_DIR, f"table_temporal_{key}.csv"), index=False)

    temp_showcase = [n for n in ["add_3", "div_2", "affine_0.9x+2"] if n in op_names]
    plot_temporal_comparison(temp_dfs, temp_showcase, "fig_temporal_comparison.png")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)

    # Aggregate: mean MAE across all smooth ops
    agg_rows = []
    for key in models_dict:
        df  = per_op_dfs[key]
        agg = float(df["mae_normalised"].mean())
        agg_rows.append({
            "model":          MODEL_LABELS[key],
            "n_params":       param_table[key],
            "mean_mae_indist":agg,
            "final_test_mse": te_hists[key][-1] if te_hists[key] else float("nan"),
        })
        print(f"  {MODEL_LABELS[key]:32s}  "
              f"params={param_table[key]:6,}  "
              f"mean_MAE={agg:.5f}  "
              f"final_MSE={te_hists[key][-1]:.6f}")

    pd.DataFrame(agg_rows).to_csv(
        os.path.join(OUT_DIR, "table_aggregate_summary.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed/60:.1f} min")
    print(f"  Outputs: {OUT_DIR}/")
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()