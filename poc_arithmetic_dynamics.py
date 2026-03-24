"""
Arithmetic Operations as Dynamical Systems  —  v2.1 (Research Build)
=====================================================================
Fixes over v2.0:
  - Trajectory clipping prevents float32 overflow (square, mul_3 blow up)
  - Per-operation z-score normalisation: each op normalised independently
    so wildly different scales (add_1 vs mul_3) don't corrupt the global mean
  - 'square' removed from training set (iterated squaring is pathologically
    explosive; retained in ANALYSIS_OPS for Exp A as illustrative failure)
  - Hard NaN guard in training loop: detects and skips corrupted batches
  - DLASCLS warning suppressed via float64 SVD in PE analysis
  - Soft Hankel rank now uses float64 internally to avoid rank-0 on tiny values

Architecture (unchanged from v2.0):
  OperationEncoder  ->  code in R^CODE_DIM  +  PE confidence gate
  HierarchicalDynamics  ->  4 parallel branches gated by code (not x)
    Branch 0  Additive        linear space, no nonlinearity
    Branch 1  Multiplicative  log space
    Branch 2  Smooth NL       Tanh MLP
    Branch 3  Piecewise       ReLU MLP

Requirements:
    pip install torch numpy matplotlib scikit-learn pandas
"""

from __future__ import annotations

import os
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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ── reproducibility ──────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = "v2_results"
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════
# 1.  OPERATION REGISTRY
# ═══════════════════════════════════════════════════════════════════════

TRAIN_OPS: OrderedDict = OrderedDict([
    # Level-0  additive
    ("add_1",         lambda x: x + 1.0),
    ("add_3",         lambda x: x + 3.0),
    ("add_5",         lambda x: x + 5.0),
    ("sub_2",         lambda x: x - 2.0),
    ("sub_4",         lambda x: x - 4.0),
    # Level-1  multiplicative
    ("mul_2",         lambda x: x * 2.0),
    ("mul_3",         lambda x: x * 3.0),
    ("div_2",         lambda x: x / 2.0),
    ("div_3",         lambda x: x / 3.0),
    ("sqrt",          lambda x: float(np.sqrt(abs(x)))),
    # Level-1  affine (mixed)
    ("affine_1.1x+1", lambda x: 1.1 * x + 1.0),
    ("affine_0.9x+2", lambda x: 0.9 * x + 2.0),
])

HARD_OPS: OrderedDict = OrderedDict([
    ("mod_add3_17",   lambda x: (x + 3.0) % 17.0),
    ("mod_add7_17",   lambda x: (x + 7.0) % 17.0),
])

# For Exp A only — includes square as overflow/failure illustration
ANALYSIS_OPS: OrderedDict = OrderedDict(
    list(TRAIN_OPS.items()) +
    [("square", lambda x: x ** 2.0)] +
    list(HARD_OPS.items())
)

CLIP_VAL = 1e6


# ═══════════════════════════════════════════════════════════════════════
# 2.  TRAJECTORY GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_trajectory(fn, x0: float, length: int,
                        dtype=np.float32, clip: float = CLIP_VAL) -> np.ndarray:
    traj = np.zeros(length, dtype=dtype)
    traj[0] = x0
    for i in range(1, length):
        val = fn(float(traj[i - 1]))
        val = float(np.clip(val, -clip, clip))
        if not np.isfinite(val):
            val = 0.0
        traj[i] = val
    return traj


def make_dataset(operations: OrderedDict, n_per_op: int, traj_len: int,
                 x0_range: Tuple[float, float] = (1.0, 10.0),
                 fixed_x0: Optional[float] = None,
                 clip: float = CLIP_VAL) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    trajs, labels = [], []
    for idx, (_, fn) in enumerate(operations.items()):
        for _ in range(n_per_op):
            x0 = fixed_x0 if fixed_x0 is not None else np.random.uniform(*x0_range)
            trajs.append(generate_trajectory(fn, x0, traj_len, clip=clip))
            labels.append(idx)
    return (np.array(trajs, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            list(operations.keys()))


# ═══════════════════════════════════════════════════════════════════════
# 3.  NORMALISATION  — per-operation
# ═══════════════════════════════════════════════════════════════════════

class PerOpNormaliser:
    """
    Fits separate (mu, sigma) per operation index.
    Prevents scale-diverse ops from poisoning global normalisation.
    """
    def __init__(self):
        self.stats: Dict[int, Tuple[float, float]] = {}

    def fit(self, trajs: np.ndarray, labels: np.ndarray) -> "PerOpNormaliser":
        for op_idx in np.unique(labels):
            subset = trajs[labels == op_idx]
            mu    = float(np.nanmean(subset))
            sigma = float(np.nanstd(subset)) + 1e-8
            self.stats[int(op_idx)] = (mu, sigma)
        return self

    def transform(self, trajs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        out = np.zeros_like(trajs, dtype=np.float32)
        for op_idx, (mu, sigma) in self.stats.items():
            mask = labels == op_idx
            if mask.any():
                out[mask] = (trajs[mask] - mu) / sigma
        return out

    def get(self, op_idx: int) -> Tuple[float, float]:
        return self.stats[int(op_idx)]


def split_tensors(trajs_n: np.ndarray, labels: np.ndarray, ctx_len: int,
                  train_frac: float = 0.8):
    valid   = np.all(np.isfinite(trajs_n), axis=1)
    trajs_n = trajs_n[valid]
    labels  = labels[valid]
    ctx_all = torch.from_numpy(trajs_n[:, :ctx_len]).float()
    tgt_all = torch.from_numpy(trajs_n[:, ctx_len:]).float()
    lbl_all = torch.from_numpy(labels).long()
    perm    = torch.randperm(len(ctx_all))
    split   = int(train_frac * len(ctx_all))
    tr, te  = perm[:split], perm[split:]
    return (ctx_all[tr], tgt_all[tr], lbl_all[tr],
            ctx_all[te], tgt_all[te], lbl_all[te])


# ═══════════════════════════════════════════════════════════════════════
# 4.  HANKEL / PE UTILITIES  (float64 for numerical stability)
# ═══════════════════════════════════════════════════════════════════════

def hankel_matrix(traj: np.ndarray, depth: int) -> np.ndarray:
    cols = len(traj) - depth + 1
    h = np.zeros((depth, max(cols, 1)), dtype=np.float64)
    for i in range(depth):
        h[i, :cols] = traj[i: i + cols].astype(np.float64)
    return h


def pe_analysis(traj: np.ndarray, max_order: int = 6) -> Dict:
    results = {}
    traj64  = traj.astype(np.float64)
    for order in range(1, max_order + 1):
        if len(traj64) < 2 * order:
            continue
        h = hankel_matrix(traj64, order)
        if not np.all(np.isfinite(h)):
            results[order] = {"rank": 0, "is_pe": False,
                               "singular_values": [], "condition_number": float("nan")}
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = np.linalg.svd(h, compute_uv=False)
        tol  = max(h.shape) * np.finfo(np.float64).eps * sv[0] if sv[0] > 0 else 1e-10
        rank = int(np.sum(sv > tol))
        cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-30 else float("nan")
        results[order] = {"rank": rank, "is_pe": bool(rank >= order),
                          "singular_values": sv.tolist(), "condition_number": cond}
    return results


def soft_hankel_rank(ctx: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    B, L  = ctx.shape
    depth = max(2, L // 2)
    cols  = L - depth + 1
    if cols < 1:
        return torch.ones(B, dtype=torch.float32)
    rows = [ctx[:, i: i + cols] for i in range(depth)]
    H    = torch.stack(rows, dim=1).double()
    try:
        S = torch.linalg.svdvals(H).float()
    except Exception:
        return torch.ones(B, dtype=torch.float32)
    return (S / (S + eps)).sum(dim=-1).float()


# ═══════════════════════════════════════════════════════════════════════
# 5.  ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

class OperationEncoder(nn.Module):
    def __init__(self, ctx_len: int, code_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_len, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, code_dim),
        )
        self.pe_threshold = nn.Parameter(torch.tensor(1.5))

    def forward(self, ctx: torch.Tensor):
        code       = self.net(ctx)
        soft_rank  = soft_hankel_rank(ctx)
        confidence = torch.sigmoid(soft_rank - self.pe_threshold)
        return code, confidence.unsqueeze(-1)


class HierarchicalDynamics(nn.Module):
    def __init__(self, code_dim: int, hidden: int = 64):
        super().__init__()
        inp = 1 + code_dim
        self.branch_add    = nn.Linear(inp, 1)
        self.branch_mul    = nn.Linear(inp, 1)
        self.branch_smooth = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        self.branch_pw = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(code_dim, 32), nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, z: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        inp     = torch.cat([z, code], dim=-1)
        log_z   = torch.log(torch.abs(z) + 1e-8) * torch.sign(z + 1e-12)
        log_inp = torch.cat([log_z, code], dim=-1)
        h0 = self.branch_add(inp)
        h1 = torch.exp(torch.clamp(self.branch_mul(log_inp), -20, 20))
        h2 = self.branch_smooth(inp)
        h3 = self.branch_pw(inp)
        g  = torch.softmax(self.gate(code), dim=-1)
        return g[:, 0:1]*h0 + g[:, 1:2]*h1 + g[:, 2:3]*h2 + g[:, 3:4]*h3

    def gate_weights(self, code: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(code), dim=-1)


class ArithmeticWorldModel(nn.Module):
    def __init__(self, ctx_len: int, code_dim: int):
        super().__init__()
        self.encoder  = OperationEncoder(ctx_len, code_dim)
        self.dynamics = HierarchicalDynamics(code_dim)

    def forward(self, context: torch.Tensor, n_steps: int,
                return_extras: bool = False):
        code, confidence = self.encoder(context)
        gated_code = confidence * code
        z = context[:, -1:]
        preds = []
        for _ in range(n_steps):
            z = self.dynamics(z, gated_code)
            preds.append(z)
        out = torch.cat(preds, dim=1)
        if return_extras:
            return out, confidence, self.dynamics.gate_weights(gated_code)
        return out


# ═══════════════════════════════════════════════════════════════════════
# 6.  TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_model(tr_ctx, tr_tgt, te_ctx, te_tgt,
                ctx_len, code_dim, n_pred,
                epochs=200, lr=1e-3, bs=64,
                verbose=True, label="",
                lambda_entropy=0.01):
    model   = ArithmeticWorldModel(ctx_len, code_dim)
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
            idx      = perm[i: i + bs]
            ctx_b    = tr_ctx[idx]
            tgt_b    = tr_tgt[idx]
            if not (torch.isfinite(ctx_b).all() and torch.isfinite(tgt_b).all()):
                continue
            pred, _, gate_probs = model(ctx_b, n_pred, return_extras=True)
            prediction_loss = loss_fn(pred, tgt_b)
            gate_entropy_loss = -torch.mean(
                torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=-1)
            )
            loss = prediction_loss - lambda_entropy * gate_entropy_loss
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
            te_loss = loss_fn(te_pred, te_tgt).item() if torch.isfinite(te_pred).all() else float("nan")

        tr_hist.append(ep_loss / max(nb, 1))
        te_hist.append(te_loss)
        if verbose and (ep + 1) % 50 == 0:
            tag = f"[{label}] " if label else ""
            print(f"    {tag}Epoch {ep+1:4d} | Train {tr_hist[-1]:.6f} | Test {te_loss:.6f}")

    return model, tr_hist, te_hist


# ═══════════════════════════════════════════════════════════════════════
# 7.  GLOBALS
# ═══════════════════════════════════════════════════════════════════════

CTX_LEN      = 5
PRED_LEN     = 15
TRAJ_LEN     = CTX_LEN + PRED_LEN
CODE_DIM     = 12
N_PER_OP     = 300
EPOCHS       = 250
BRANCH_NAMES = ["Additive (L0)", "Multiplicative (L1)", "Smooth NL (L2)", "Piecewise (L3)"]
BRANCH_SHORT = ["Add", "Mul", "Smooth", "PW"]


# ═══════════════════════════════════════════════════════════════════════
# 8.  EXP A — PE Characterisation
# ═══════════════════════════════════════════════════════════════════════

def exp_A_pe_characterisation(ops: OrderedDict):
    print("\n" + "="*70)
    print("  EXP A — PE CHARACTERISATION & HANKEL SIGNATURE")
    print("="*70)

    rows, sv_data = [], {}
    for name, fn in ops.items():
        x0_a   = 2.0  if name == "square" else 5.0
        clip_a = 1e4  if name == "square" else CLIP_VAL
        traj   = generate_trajectory(fn, x0_a, 60, dtype=np.float64, clip=clip_a)
        res    = pe_analysis(traj, max_order=6)
        order_sv = 3 if 3 in res else (max(res.keys()) if res else 1)
        sv_data[name] = res.get(order_sv, {}).get("singular_values", [])[:6]
        row = {"operation": name}
        for order, r in res.items():
            row[f"rank_ord{order}"] = r["rank"]
            row[f"PE_ord{order}"]   = r["is_pe"]
        row["cond_num_ord3"] = res.get(3, {}).get("condition_number", float("nan"))
        rows.append(row)
        pe_str = " | ".join(f"ord{k}:{'V' if v['is_pe'] else 'X'}(r={v['rank']})"
                            for k, v in res.items())
        print(f"  {name:22s}  {pe_str}  cond={row['cond_num_ord3']:.2e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "tableA_pe_characterisation.csv"), index=False)

    # Heatmap
    op_names  = list(sv_data.keys())
    sv_matrix = np.array([(sv_data[n][:6] + [0.0]*6)[:6] for n in op_names], dtype=float)
    fig, ax   = plt.subplots(figsize=(11, 7))
    im = ax.imshow(np.log1p(np.abs(sv_matrix)), aspect="auto", cmap="viridis")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"s{i+1}" for i in range(6)])
    ax.set_yticks(range(len(op_names)))
    ax.set_yticklabels(op_names, fontsize=9)
    plt.colorbar(im, ax=ax, label="log(1+|s|)")
    ax.set_title("Hankel Singular Value Spectrum by Operation\n"
                 "Steep drop after s1/s2 -> PE-identifiable  |  Flat -> high-dim or periodic")
    plt.tight_layout()
    savefig("figA_hankel_sv_heatmap.png")

    # SV decay per op
    ncols = 5
    nrows = (len(ops) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3))
    axes = axes.flatten()
    for i, (name, fn) in enumerate(ops.items()):
        ax    = axes[i]
        x0_a  = 2.0  if name == "square" else 5.0
        clip_a = 1e4 if name == "square" else CLIP_VAL
        traj  = generate_trajectory(fn, x0_a, 60, dtype=np.float64, clip=clip_a)
        h     = hankel_matrix(traj, 6)
        if np.all(np.isfinite(h)) and h.shape[1] > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = np.linalg.svd(h, compute_uv=False)
            ax.semilogy(range(1, len(sv)+1), sv + 1e-30, "o-", markersize=4)
        else:
            ax.text(0.5, 0.5, "overflow/nan", transform=ax.transAxes,
                    ha="center", color="red")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Index", fontsize=7)
        ax.set_ylabel("s", fontsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Singular Value Decay of Order-6 Hankel Matrix\n"
                 "Steep decay -> low intrinsic dimension (easy to identify)", fontsize=11)
    plt.tight_layout()
    savefig("figA2_sv_decay.png")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 9.  EXP B — PE Necessity
# ═══════════════════════════════════════════════════════════════════════

def exp_B_pe_necessity():
    print("\n" + "="*70)
    print("  EXP B — PE NECESSITY (SIZE-CONTROLLED)")
    print("="*70)

    pe_ops = OrderedDict((k, v) for k, v in TRAIN_OPS.items()
                         if k.startswith("add") or k.startswith("sub"))
    N_FIXED = 300
    colors  = ["#d9534f", "#f0ad4e", "#5cb85c"]

    # Shared test set
    te_raw, te_lbl, _ = make_dataset(pe_ops, 100, TRAJ_LEN, x0_range=(1.0, 10.0))
    norm_te = PerOpNormaliser().fit(te_raw, te_lbl)
    te_n    = norm_te.transform(te_raw, te_lbl)
    valid   = np.all(np.isfinite(te_n), axis=1)
    te_n, te_lbl_v = te_n[valid], te_lbl[valid]
    te_ctx  = torch.from_numpy(te_n[:, :CTX_LEN]).float()
    te_tgt  = torch.from_numpy(te_n[:, CTX_LEN:]).float()

    conditions = [
        ("FIXED  (x0=5.0)", dict(fixed_x0=5.0,  x0_range=(1.0, 10.0))),
        ("LOW    (+-0.1)",  dict(fixed_x0=None,  x0_range=(4.9, 5.1))),
        ("HIGH   ([1,10])", dict(fixed_x0=None,  x0_range=(1.0, 10.0))),
    ]
    results, hists = {}, {}

    for (lbl, kwargs), color in zip(conditions, colors):
        print(f"\n  Condition: {lbl}")
        raw, lbl_arr, _ = make_dataset(
            pe_ops, N_FIXED, TRAJ_LEN,
            fixed_x0=kwargs.get("fixed_x0"),
            x0_range=kwargs.get("x0_range", (1.0, 10.0)),
        )
        norm_cond = PerOpNormaliser().fit(raw, lbl_arr)
        raw_n     = norm_cond.transform(raw, lbl_arr)
        valid     = np.all(np.isfinite(raw_n), axis=1)
        raw_n, lbl_arr = raw_n[valid], lbl_arr[valid]
        ctx = torch.from_numpy(raw_n[:, :CTX_LEN]).float()
        tgt = torch.from_numpy(raw_n[:, CTX_LEN:]).float()
        perm = torch.randperm(len(ctx))
        s    = int(0.8 * len(ctx))
        m, _, te_h = train_model(
            ctx[perm[:s]], tgt[perm[:s]], te_ctx, te_tgt,
            CTX_LEN, CODE_DIM, PRED_LEN,
            epochs=EPOCHS, verbose=True, label=lbl.strip(),
        )
        m.eval()
        with torch.no_grad():
            final = nn.MSELoss()(m(te_ctx, PRED_LEN), te_tgt).item()
        results[lbl] = final
        hists[lbl]   = te_h
        print(f"  -> Final test MSE: {final:.6f}")

    labels_list = list(results.keys())
    vals        = list(results.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels_list, vals, color=colors, width=0.5)
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.05, f"{v:.5f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Final Test MSE")
    ax.set_yscale("log")
    ax.set_title("PE Necessity Experiment\n"
                 "(Same N=300 — only x0 diversity varies)\n"
                 "Gap between FIXED and HIGH isolates PE as causal")
    plt.tight_layout()
    savefig("figB_pe_necessity.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    for (lbl, _), color in zip(conditions, colors):
        h = [v for v in hists[lbl] if np.isfinite(v)]
        if h:
            ax.semilogy(h, label=lbl.strip(), color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE")
    ax.set_title("Learning Curves by PE Condition\n(same N, different x0 diversity)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("figB2_pe_necessity_curves.png")

    pd.DataFrame({"condition": labels_list, "final_test_mse": vals}).to_csv(
        os.path.join(OUT_DIR, "tableB_pe_necessity.csv"), index=False)
    return results


# ═══════════════════════════════════════════════════════════════════════
# 10. EXP C — Train Main Model
# ═══════════════════════════════════════════════════════════════════════

def exp_C_train_main():
    print("\n" + "="*70)
    print("  EXP C — TRAIN MAIN MODEL")
    print("="*70)

    trajs, labels, op_names = make_dataset(TRAIN_OPS, N_PER_OP, TRAJ_LEN)
    norm    = PerOpNormaliser().fit(trajs, labels)
    trajs_n = norm.transform(trajs, labels)

    (tr_ctx, tr_tgt, tr_lbl,
     te_ctx, te_tgt, te_lbl) = split_tensors(trajs_n, labels, CTX_LEN)
    print(f"  Train {len(tr_ctx)} | Test {len(te_ctx)} | Ops {len(op_names)}")

    model, tr_h, te_h = train_model(
        tr_ctx, tr_tgt, te_ctx, te_tgt,
        CTX_LEN, CODE_DIM, PRED_LEN,
        epochs=EPOCHS, verbose=True, label="main",
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(tr_h, label="Train", alpha=0.8)
    ax.semilogy(te_h, label="Test",  alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training Curves — Hierarchical Dynamics Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("figC_training_curves.png")

    return model, norm, op_names, tr_ctx, tr_tgt, te_ctx, te_tgt, te_lbl


# ═══════════════════════════════════════════════════════════════════════
# 11. EXP D — Operation Code Clustering
# ═══════════════════════════════════════════════════════════════════════

def exp_D_code_clustering(model, te_ctx, te_lbl, op_names):
    print("\n" + "="*70)
    print("  EXP D — OPERATION CODE CLUSTERING")
    print("="*70)

    model.eval()
    with torch.no_grad():
        codes, confs = model.encoder(te_ctx)
        codes = codes.numpy()
        confs = confs.squeeze().numpy()
        lbls  = te_lbl.numpy()

    valid   = np.all(np.isfinite(codes), axis=1)
    codes_v = codes[valid]
    confs_v = confs[valid]
    lbls_v  = lbls[valid]

    sil = silhouette_score(codes_v, lbls_v) if len(np.unique(lbls_v)) > 1 else 0.0
    print(f"  Silhouette score   : {sil:.4f}")
    print(f"  Mean PE confidence : {confs_v.mean():.4f} +/- {confs_v.std():.4f}")

    tsne = TSNE(n_components=2, perplexity=min(30, len(codes_v)//4), random_state=SEED)
    c2d  = tsne.fit_transform(codes_v)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.cm.get_cmap("tab20", len(op_names))

    ax = axes[0]
    for i, nm in enumerate(op_names):
        m = lbls_v == i
        if m.any():
            ax.scatter(c2d[m, 0], c2d[m, 1], label=nm, alpha=0.6, s=20, color=cmap(i))
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"t-SNE of Operation Codes\nSilhouette = {sil:.3f}")

    ax = axes[1]
    sc = ax.scatter(c2d[:, 0], c2d[:, 1], c=confs_v,
                    cmap="RdYlGn", s=20, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="PE Confidence")
    ax.set_title("Codes Coloured by PE Confidence\n(Green=high, Red=low)")

    plt.suptitle("Learned Operation Code Space", fontsize=13)
    plt.tight_layout()
    savefig("figD_code_clustering.png")

    conf_rows = [{"operation": op_names[i],
                  "mean_confidence": float(confs_v[lbls_v == i].mean()),
                  "std_confidence":  float(confs_v[lbls_v == i].std())}
                 for i in range(len(op_names)) if (lbls_v == i).any()]
    df = pd.DataFrame(conf_rows)
    df.to_csv(os.path.join(OUT_DIR, "tableD_pe_confidence.csv"), index=False)
    print(df.to_string(index=False))
    return sil, codes_v, confs_v, lbls_v


# ═══════════════════════════════════════════════════════════════════════
# 12. EXP E — Branch Gate Analysis
# ═══════════════════════════════════════════════════════════════════════

def exp_E_branch_gates(model, te_ctx, te_lbl, op_names):
    print("\n" + "="*70)
    print("  EXP E — BRANCH GATE ANALYSIS (INTERPRETABILITY)")
    print("="*70)

    model.eval()
    gate_matrix = np.zeros((len(op_names), 4))
    rows = []

    with torch.no_grad():
        for i, nm in enumerate(op_names):
            m = te_lbl == i
            if not m.any():
                continue
            _, _, gates = model(te_ctx[m], 1, return_extras=True)
            mg = gates.mean(dim=0).numpy()
            gate_matrix[i] = mg
            dom = BRANCH_NAMES[int(np.argmax(mg))]
            row = {"operation": nm, "dominant_branch": dom}
            for j, bn in enumerate(BRANCH_NAMES):
                row[bn] = float(mg[j])
            rows.append(row)
            print(f"  {nm:22s}  dominant={dom:22s}  "
                  f"gates={[f'{g:.3f}' for g in mg]}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "tableE_branch_gates.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(gate_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(BRANCH_NAMES, rotation=15, ha="right", fontsize=9)
    ax.set_yticks(range(len(op_names)))
    ax.set_yticklabels(op_names, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean Gate Weight")
    for ii in range(len(op_names)):
        for jj in range(4):
            ax.text(jj, ii, f"{gate_matrix[ii,jj]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if gate_matrix[ii,jj] < 0.6 else "white")
    ax.set_title("Branch Gate Weights by Operation\n"
                 "Key: additive ops -> B0, multiplicative -> B1")
    plt.tight_layout()
    savefig("figE_branch_gates.png")


# ═══════════════════════════════════════════════════════════════════════
# 13. EXP F — Zero-Shot Transfer
# ═══════════════════════════════════════════════════════════════════════

def exp_F_zero_shot_transfer(model, norm: PerOpNormaliser, op_names):
    print("\n" + "="*70)
    print("  EXP F — ZERO-SHOT TRANSFER")
    print("="*70)

    model.eval()
    enc_x0   = 5.0
    exec_x0s = [20.0, 50.0, 200.0, 1000.0]
    rows     = []

    for op_idx, (name, fn) in enumerate(TRAIN_OPS.items()):
        mu, sigma = norm.get(op_idx)
        ref_traj  = generate_trajectory(fn, enc_x0, TRAJ_LEN)
        ref_n     = (ref_traj - mu) / sigma
        if not np.all(np.isfinite(ref_n)):
            continue
        ref_ctx = torch.from_numpy(ref_n[:CTX_LEN]).unsqueeze(0).float()
        with torch.no_grad():
            ref_code, ref_conf = model.encoder(ref_ctx)

        row = {"operation": name, "enc_x0": enc_x0,
               "ref_confidence": float(ref_conf.item())}

        for x0_exec in exec_x0s:
            true_traj = generate_trajectory(fn, x0_exec, TRAJ_LEN)
            true_n    = (true_traj - mu) / sigma
            if not np.all(np.isfinite(true_n)):
                row[f"mae_full_x0{int(x0_exec)}"]     = float("nan")
                row[f"mae_injected_x0{int(x0_exec)}"] = float("nan")
                continue
            exec_ctx = torch.from_numpy(true_n[:CTX_LEN]).unsqueeze(0).float()

            with torch.no_grad():
                pred_full_n = model(exec_ctx, PRED_LEN).squeeze(0).numpy()
            pred_full = pred_full_n * sigma + mu

            zi = torch.tensor([[true_n[CTX_LEN - 1]]])
            gated = ref_conf * ref_code
            preds_inj = []
            with torch.no_grad():
                for _ in range(PRED_LEN):
                    zi = model.dynamics(zi, gated)
                    preds_inj.append(float(zi.item()))
            pred_inj = np.array(preds_inj) * sigma + mu

            true_fut = true_traj[CTX_LEN:]
            row[f"mae_full_x0{int(x0_exec)}"]     = float(np.nanmean(np.abs(pred_full - true_fut)))
            row[f"mae_injected_x0{int(x0_exec)}"] = float(np.nanmean(np.abs(pred_inj  - true_fut)))

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "tableF_zero_shot_transfer.csv"), index=False)
    print(df.to_string(index=False))

    inj_cols = [f"mae_injected_x0{int(x)}" for x in exec_x0s]
    mat = df[inj_cols].values.astype(float)
    op_list = df["operation"].tolist()

    fig, ax = plt.subplots(figsize=(9, max(4, len(op_list)*0.5+1)))
    im = ax.imshow(np.log1p(np.abs(np.nan_to_num(mat, nan=0))),
                   aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(exec_x0s)))
    ax.set_xticklabels([f"x0={int(x)}" for x in exec_x0s])
    ax.set_yticks(range(len(op_list)))
    ax.set_yticklabels(op_list, fontsize=8)
    for ii in range(len(op_list)):
        for jj in range(len(exec_x0s)):
            v = mat[ii, jj]
            ax.text(jj, ii, f"{v:.2f}" if np.isfinite(v) else "nan",
                    ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="log(1+MAE)")
    ax.set_title("Zero-Shot Transfer MAE (Injected Code)\n"
                 "Code encoded from x0=5, executed from OOD x0")
    plt.tight_layout()
    savefig("figF_zero_shot_transfer.png")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 14. EXP G — Failure Mode Characterisation
# ═══════════════════════════════════════════════════════════════════════

def exp_G_failure_modes(norm: PerOpNormaliser):
    print("\n" + "="*70)
    print("  EXP G — FAILURE MODE CHARACTERISATION")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    showcase = [
        ("add_3",       TRAIN_OPS["add_3"],      "Smooth (additive)"),
        ("mod_add3_17", HARD_OPS["mod_add3_17"], "Hard (modular)"),
        ("square",      ANALYSIS_OPS["square"],  "Explosive (square, x0=2)"),
    ]
    for col, (name, fn, label) in enumerate(showcase):
        x0_g   = 2.0  if name == "square" else 5.0
        clip_g = 1e4  if name == "square" else CLIP_VAL
        traj   = generate_trajectory(fn, x0_g, 80, dtype=np.float64, clip=clip_g)
        ax = axes[0, col]
        ax.plot(np.arange(len(traj)), traj, ".-", markersize=4)
        ax.set_title(f"{name}\n({label})", fontsize=9)
        ax.set_xlabel("Step k")
        ax.set_ylabel("x_k")
        ax.grid(True, alpha=0.3)
        ax = axes[1, col]
        for depth in [3, 5, 8]:
            h = hankel_matrix(traj[:30], depth)
            if np.all(np.isfinite(h)) and h.shape[1] > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sv = np.linalg.svd(h, compute_uv=False)
                ax.semilogy(range(1, len(sv)+1), sv + 1e-30,
                            "o-", label=f"depth={depth}", markersize=4)
        ax.set_title(f"Hankel SV: {name}", fontsize=9)
        ax.set_xlabel("SV index")
        ax.set_ylabel("s (log)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Failure Mode Analysis: Hankel Structure\n"
                 "Smooth: steep decay -> identifiable | "
                 "Modular: flat -> periodic | Square: overflow", fontsize=10)
    plt.tight_layout()
    savefig("figG_failure_mode_hankel.png")

    # Soft rank per op
    print("\n  Soft Hankel rank scores:")
    rows = []
    for op_idx, (name, fn) in enumerate(ANALYSIS_OPS.items()):
        op_type = ("modular" if "mod" in name else
                   "explosive" if name == "square" else "smooth")
        try:
            mu, sigma = norm.get(min(op_idx, len(TRAIN_OPS)-1))
        except Exception:
            mu, sigma = 5.0, 1.0
        batch = []
        for x0 in np.linspace(1, 10, 40):
            t = generate_trajectory(fn, float(x0), TRAJ_LEN,
                                    clip=1e4 if name == "square" else CLIP_VAL)
            t_n = (t - mu) / (sigma + 1e-8)
            if np.all(np.isfinite(t_n)):
                batch.append(t_n[:CTX_LEN])
        if not batch:
            rows.append({"operation": name, "mean_soft_rank": float("nan"), "type": op_type})
            continue
        ctx_t = torch.from_numpy(np.array(batch, dtype=np.float32))
        sr    = soft_hankel_rank(ctx_t).mean().item()
        rows.append({"operation": name, "mean_soft_rank": sr, "type": op_type})
        print(f"    {name:22s}  soft_rank={sr:.3f}  [{op_type}]")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "tableG_soft_rank.csv"), index=False)

    finite_df = df[df["mean_soft_rank"].apply(np.isfinite)]
    cmap_g = {"smooth": "#5cb85c", "modular": "#d9534f", "explosive": "#777777"}
    colors_g = [cmap_g.get(t, "#aaaaaa") for t in finite_df["type"]]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.barh(finite_df["operation"], finite_df["mean_soft_rank"], color=colors_g)
    ax.axvline(1.5, color="black", linestyle="--", label="PE threshold ~1.5")
    ax.set_xlabel("Mean Soft Hankel Rank")
    ax.set_title("Soft Hankel Rank by Operation\n"
                 "Green=smooth, Red=modular, Grey=explosive")
    ax.legend()
    plt.tight_layout()
    savefig("figG2_soft_rank_by_op.png")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 15. EXP H — Code Space Algebra
# ═══════════════════════════════════════════════════════════════════════

def exp_H_code_algebra(model, norm: PerOpNormaliser, op_names):
    print("\n" + "="*70)
    print("  EXP H — CODE SPACE ALGEBRA TEST")
    print("="*70)

    model.eval()
    ref_codes = {}
    for op_idx, (name, fn) in enumerate(TRAIN_OPS.items()):
        mu, sigma = norm.get(op_idx)
        batch = []
        for x0 in np.random.uniform(1, 10, 80):
            t = generate_trajectory(fn, float(x0), TRAJ_LEN)
            t_n = (t - mu) / sigma
            if np.all(np.isfinite(t_n)):
                batch.append(t_n[:CTX_LEN])
        if not batch:
            continue
        ctx_t = torch.from_numpy(np.array(batch, dtype=np.float32))
        with torch.no_grad():
            c, _ = model.encoder(ctx_t)
            ref_codes[name] = c.mean(dim=0)

    interp_corr = None
    if "add_1" in ref_codes and "add_5" in ref_codes:
        c1 = ref_codes["add_1"]
        c5 = ref_codes["add_5"]
        op_idx_1 = list(TRAIN_OPS.keys()).index("add_1")
        mu1, sigma1 = norm.get(op_idx_1)
        alphas = np.linspace(0, 1, 25)
        steps_pred = []
        x0_norm = (5.0 - mu1) / sigma1
        for alpha in alphas:
            c_interp = ((1 - alpha) * c1 + alpha * c5).unsqueeze(0)
            z = torch.tensor([[x0_norm]])
            with torch.no_grad():
                z_next = model.dynamics(z, c_interp)
            step_real = (float(z_next.item()) - x0_norm) * sigma1
            steps_pred.append(step_real)
        expected = 1.0 + alphas * 4.0
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(alphas, steps_pred, "b.-", markersize=5,
                label="Model step at interpolated code")
        ax.plot(alphas, expected,   "r--", label="Expected (linear 1->5)")
        ax.set_xlabel("Interpolation alpha  (0=add_1, 1=add_5)")
        ax.set_ylabel("Predicted step size")
        ax.set_title("Code Space Algebra: Interpolating code(add_1) -> code(add_5)\n"
                     "Linear code interpolation should produce linear step interpolation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        savefig("figH_code_interpolation.png")
        finite_mask = [np.isfinite(s) for s in steps_pred]
        if sum(finite_mask) > 3:
            sp  = [s for s, f in zip(steps_pred, finite_mask) if f]
            ex  = [e for e, f in zip(expected,   finite_mask) if f]
            interp_corr = float(np.corrcoef(sp, ex)[0, 1])
            print(f"  Interpolation linearity (Pearson r): {interp_corr:.4f}")

    # Unseen composition
    print("\n  Unseen composition test:")
    composed = OrderedDict([
        ("add_4", lambda x: x + 4.0),
        ("add_6", lambda x: x + 6.0),
        ("add_8", lambda x: x + 8.0),
    ])
    op_idx_ref = list(TRAIN_OPS.keys()).index("add_1")
    mu_ref, sigma_ref = norm.get(op_idx_ref)
    comp_rows = []
    for comp_name, comp_fn in composed.items():
        batch_c, batch_true = [], []
        for x0 in np.random.uniform(1, 10, 60):
            t = generate_trajectory(comp_fn, float(x0), TRAJ_LEN)
            t_n = (t - mu_ref) / sigma_ref
            if np.all(np.isfinite(t_n)):
                batch_c.append(t_n[:CTX_LEN])
                batch_true.append(t_n[CTX_LEN:])
        if not batch_c:
            continue
        ctx_t  = torch.from_numpy(np.array(batch_c,    dtype=np.float32))
        true_t = torch.from_numpy(np.array(batch_true, dtype=np.float32))
        with torch.no_grad():
            pred    = model(ctx_t, PRED_LEN)
            comp_c, _ = model.encoder(ctx_t)
            comp_c  = comp_c.mean(dim=0)
            mae_n   = torch.abs(pred - true_t).mean().item()
        dists   = {k: float(torch.norm(comp_c - v).item()) for k, v in ref_codes.items()}
        nearest = min(dists, key=dists.get)
        top3    = sorted(dists.items(), key=lambda x: x[1])[:3]
        print(f"  {comp_name:8s}  MAE(norm)={mae_n:.5f}  nearest={nearest}  "
              f"top3={[(n, round(d,3)) for n,d in top3]}")
        comp_rows.append({"unseen_op": comp_name, "mae_norm": mae_n,
                          "nearest_known": nearest,
                          **{f"dist_{n}": d for n,d in top3}})
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(OUT_DIR, "tableH_code_algebra.csv"), index=False)
    return interp_corr


# ═══════════════════════════════════════════════════════════════════════
# 16. EXP I — Rollouts
# ═══════════════════════════════════════════════════════════════════════

def exp_I_rollouts(model, norm: PerOpNormaliser, op_names):
    print("\n" + "="*70)
    print("  EXP I — TRAJECTORY ROLLOUT PLOTS")
    print("="*70)

    all_rollout_ops = OrderedDict(list(TRAIN_OPS.items()) + list(HARD_OPS.items()))
    n_ops  = len(all_rollout_ops)
    ncols  = 4
    nrows  = (n_ops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3.5))
    axes = axes.flatten()
    model.eval()
    rows = []

    for i, (name, fn) in enumerate(all_rollout_ops.items()):
        ax = axes[i]
        op_idx = list(TRAIN_OPS.keys()).index(name) if name in TRAIN_OPS else 0
        mu, sigma = norm.get(op_idx)
        true  = generate_trajectory(fn, 5.0, CTX_LEN + 20)
        true_n = (true - mu) / sigma
        if not np.all(np.isfinite(true_n)):
            ax.set_title(f"{name}\n(overflow)", fontsize=8)
            ax.axis("off")
            continue
        c_t = torch.from_numpy(true_n[:CTX_LEN]).unsqueeze(0).float()
        with torch.no_grad():
            p_n, conf, gates = model(c_t, 20, return_extras=True)
            p_n   = p_n.squeeze(0).numpy()
            conf  = float(conf.item())
            gates = gates.squeeze(0).numpy()
        pred = p_n * sigma + mu
        t_ctx = np.arange(CTX_LEN)
        t_fut = np.arange(CTX_LEN, CTX_LEN + 20)
        mae   = float(np.nanmean(np.abs(pred - true[CTX_LEN:])))
        dom   = BRANCH_SHORT[int(np.argmax(gates))]
        ax.plot(t_ctx, true[:CTX_LEN], "k.-", label="Context", markersize=4)
        ax.plot(t_fut, true[CTX_LEN:], "b.-", label="True",    markersize=4, alpha=0.7)
        ax.plot(t_fut, pred,           "r.--",label="Predicted",markersize=4, alpha=0.8)
        ax.axvline(CTX_LEN - 0.5, color="gray", ls=":", alpha=0.5)
        ax.set_title(f"{name}\nMAE={mae:.3f}  conf={conf:.2f}  br={dom}", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)
        rows.append({"operation": name, "mae_rollout": mae,
                     "pe_confidence": conf, "dominant_branch": dom})

    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Trajectory Rollouts (x0=5, 20 steps; trained on 15)\n"
                 "MAE | PE confidence | dominant branch", fontsize=11)
    plt.tight_layout()
    savefig("figI_rollouts.png")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "tableI_rollout_summary.csv"), index=False)
    print(df.to_string(index=False))
    return df


# ═══════════════════════════════════════════════════════════════════════
# 17. EXP J — Extrapolation Stress Test
# ═══════════════════════════════════════════════════════════════════════

def exp_J_extrapolation(model, norm: PerOpNormaliser):
    print("\n" + "="*70)
    print("  EXP J — EXTRAPOLATION STRESS TEST")
    print("="*70)
    model.eval()

    print("  J1 — Temporal:")
    rows_temp = []
    for op_idx, (name, fn) in enumerate(TRAIN_OPS.items()):
        mu, sigma = norm.get(op_idx)
        true_long = generate_trajectory(fn, 5.0, CTX_LEN + 4*PRED_LEN)
        true_n    = (true_long - mu) / sigma
        if not np.all(np.isfinite(true_n)):
            continue
        c_t = torch.from_numpy(true_n[:CTX_LEN]).unsqueeze(0).float()
        with torch.no_grad():
            p_n = model(c_t, 4*PRED_LEN).squeeze(0).numpy()
        pred = p_n * sigma + mu
        fut  = true_long[CTX_LEN:]
        row  = {"operation": name}
        for seg, lo, hi in [("steps_01_15", 0, PRED_LEN),
                             ("steps_16_30", PRED_LEN, 2*PRED_LEN),
                             ("steps_31_60", 2*PRED_LEN, 4*PRED_LEN)]:
            row[seg] = float(np.nanmean(np.abs(pred[lo:hi] - fut[lo:hi])))
        rows_temp.append(row)
        print(f"    {name:22s}  "
              f"1-15:{row['steps_01_15']:.4f}  "
              f"16-30:{row['steps_16_30']:.4f}  "
              f"31-60:{row['steps_31_60']:.4f}")
    df_temp = pd.DataFrame(rows_temp)
    df_temp.to_csv(os.path.join(OUT_DIR, "tableJ1_temporal_extrap.csv"), index=False)

    # Showcase plot
    showcase = [n for n in TRAIN_OPS if n in ["add_3", "mul_2", "affine_1.1x+1", "sqrt"]]
    if showcase:
        fig, axes = plt.subplots(1, len(showcase), figsize=(len(showcase)*5, 4))
        if len(showcase) == 1:
            axes = [axes]
        for ax, name in zip(axes, showcase):
            op_idx = list(TRAIN_OPS.keys()).index(name)
            mu, sigma = norm.get(op_idx)
            fn = TRAIN_OPS[name]
            true_long = generate_trajectory(fn, 5.0, CTX_LEN + 4*PRED_LEN)
            true_n    = (true_long - mu) / sigma
            c_t = torch.from_numpy(true_n[:CTX_LEN]).unsqueeze(0).float()
            with torch.no_grad():
                p_n = model(c_t, 4*PRED_LEN).squeeze(0).numpy()
            pred  = p_n * sigma + mu
            t_ctx = np.arange(CTX_LEN)
            t_fut = np.arange(CTX_LEN, CTX_LEN + 4*PRED_LEN)
            ax.plot(t_ctx, true_long[:CTX_LEN], "k.-", label="Context")
            ax.plot(t_fut, true_long[CTX_LEN:], "b.-", label="True", alpha=0.7)
            ax.plot(t_fut, pred, "r--", label="Pred", alpha=0.8)
            ax.axvspan(CTX_LEN, CTX_LEN+PRED_LEN, alpha=0.08, color="green")
            ax.axvspan(CTX_LEN+PRED_LEN, CTX_LEN+4*PRED_LEN, alpha=0.08, color="orange")
            ax.set_title(name, fontsize=9)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
        plt.suptitle("Temporal Extrapolation (green=train, orange=beyond)", fontsize=11)
        plt.tight_layout()
        savefig("figJ_temporal_extrap.png")

    print("\n  J2 — OOD x0:")
    ood_x0s = [0.5, 20.0, 100.0, 500.0]
    rows_ood = []
    for op_idx, (name, fn) in enumerate(TRAIN_OPS.items()):
        mu, sigma = norm.get(op_idx)
        row = {"operation": name}
        for x0 in ood_x0s:
            true   = generate_trajectory(fn, x0, TRAJ_LEN)
            true_n = (true - mu) / sigma
            if not np.all(np.isfinite(true_n)):
                row[f"x0_{x0}"] = float("nan")
                continue
            c_t = torch.from_numpy(true_n[:CTX_LEN]).unsqueeze(0).float()
            with torch.no_grad():
                p_n = model(c_t, PRED_LEN).squeeze(0).numpy()
            pred = p_n * sigma + mu
            row[f"x0_{x0}"] = float(np.nanmean(np.abs(pred - true[CTX_LEN:])))
        rows_ood.append(row)
        print(f"    {name:22s}  "
              + "  ".join(f"x0={x}->{row.get(f'x0_{x}', float('nan')):.3f}"
                          for x in ood_x0s))
    pd.DataFrame(rows_ood).to_csv(
        os.path.join(OUT_DIR, "tableJ2_ood_extrap.csv"), index=False)
    return df_temp, pd.DataFrame(rows_ood)


# ═══════════════════════════════════════════════════════════════════════
# 18. SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def write_summary(results: dict):
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    lines = [
        "Arithmetic Operations as Dynamical Systems v2.1",
        "="*55, "",
        "Architecture:",
        "  OperationEncoder  -> code R^12 + PE confidence gate",
        "  PE gate           = sigmoid(soft_Hankel_rank(ctx) - theta)",
        "  HierarchicalDynamics  4 branches gated by CODE not x",
        "    B0 Additive (linear)  B1 Multiplicative (log-space)",
        "    B2 Smooth NL (Tanh)   B3 Piecewise (ReLU)", "",
        "Key results:",
    ]
    for k, v in results.items():
        lines.append(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    lines += ["", "Outputs in v2_results/:"]
    for fig in ["figA-figJ  (figures)", "tableA-tableJ  (CSV data)",
                "summary_report.txt"]:
        lines.append(f"  {fig}")
    path = os.path.join(OUT_DIR, "summary_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# 19. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("="*70)
    print("  ARITHMETIC DYNAMICS v2.1  —  Research Build")
    print("="*70)
    print(f"  Output: {OUT_DIR}/")
    print(f"  Train ops: {len(TRAIN_OPS)} | Hard ops: {len(HARD_OPS)}")
    print(f"  CTX={CTX_LEN}  PRED={PRED_LEN}  CODE_DIM={CODE_DIM}  "
          f"N_PER_OP={N_PER_OP}  EPOCHS={EPOCHS}")

    summary = {}

    df_A = exp_A_pe_characterisation(ANALYSIS_OPS)
    if "PE_ord3" in df_A.columns:
        summary["ops_PE_ord3"] = int(df_A["PE_ord3"].sum())

    pe_nec = exp_B_pe_necessity()
    summary["PE_necessity_FIXED_mse"] = pe_nec.get("FIXED  (x0=5.0)", float("nan"))
    summary["PE_necessity_HIGH_mse"]  = pe_nec.get("HIGH   ([1,10])", float("nan"))

    (model, norm, op_names,
     tr_ctx, tr_tgt, te_ctx, te_tgt, te_lbl) = exp_C_train_main()

    sil, codes, confs, lbls = exp_D_code_clustering(model, te_ctx, te_lbl, op_names)
    summary["silhouette_score"]   = sil
    summary["mean_pe_confidence"] = float(confs.mean())

    exp_E_branch_gates(model, te_ctx, te_lbl, op_names)
    exp_F_zero_shot_transfer(model, norm, op_names)
    exp_G_failure_modes(norm)

    corr = exp_H_code_algebra(model, norm, op_names)
    if corr is not None:
        summary["code_algebra_linearity_r"] = corr

    df_I = exp_I_rollouts(model, norm, op_names)
    smooth_mae = df_I[~df_I["operation"].str.startswith("mod")]["mae_rollout"].mean()
    summary["mean_rollout_mae_smooth"] = float(smooth_mae)

    exp_J_extrapolation(model, norm)
    write_summary(summary)

    print(f"\n  Total runtime: {(time.time()-t0)/60:.1f} min")
    print("="*70)
    print(f"  DONE — check {OUT_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
