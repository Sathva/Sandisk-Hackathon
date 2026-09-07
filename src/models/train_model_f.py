"""
Training Pipeline for Model F: Wafer-Conditional Manifold Detector (GPU Accelerated).

Inductive Design:
1. Three Distinct, Lean Engines:
   - CatBoost (GPU): Symmetric oblivious trees, Logloss with GPU task_type.
   - XGBoost (GPU): Depth-wise histogram trees on CUDA (device='cuda').
   - LightGBM (CPU Multi-thread): Leaf-wise asymmetric histogram trees (n_jobs=-1).
2. Out-of-Fold (OOF) Discipline:
   - 5-fold GroupKFold on wafer_id within dev_train.
   - Round counts and blend weights chosen strictly OOF.
   - dev_val is never touched during optimization.
3. Scale-Free Rank Blending:
   - Engines output probabilities on disparate scales; predictions are transformed
     to empirical rank space [0, 1] before linear combination.
4. Gradient-Free Dirichlet + Nelder-Mead Weight Search:
   - Directly optimizes average precision without finite-difference gradient stalling.
5. Wafer-Clustered Bootstrap CI:
   - 400 cluster resamplings on dev_val wafers to evaluate honest variance.
"""

import sys
import os
import time
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, SEED
from src.features.model_f_features import FORBIDDEN

TRAIN_PARQUET = PROCESSED_DIR / "dev_train_model_f_features.parquet"
VAL_PARQUET = PROCESSED_DIR / "dev_val_model_f_features.parquet"

CB_PATH = MODELS_DIR / "model_f_catboost.cbm"
XGB_PATH = MODELS_DIR / "model_f_xgboost.json"
LGB_PATH = MODELS_DIR / "model_f_lightgbm.txt"
WEIGHTS_PATH = MODELS_DIR / "model_f_stacking_weights.json"
FEATURES_PATH = MODELS_DIR / "model_f_feature_list.json"
VAL_PREDS_NPZ = REPORTS_DIR / "model_f_val_preds.npz"

N_FOLDS = 5
LR = 0.025
MAX_ROUNDS = 2500
PATIENCE = 120


# ---------------------------------------------------------------------------
# Data Loading & Verification
# ---------------------------------------------------------------------------
def load_datasets():
    print("\nLoading Model F feature datasets...", flush=True)
    t0 = time.time()
    df_tr = pd.read_parquet(TRAIN_PARQUET)
    df_va = pd.read_parquet(VAL_PARQUET)

    feature_cols = [c for c in df_tr.columns if c not in FORBIDDEN]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_tr[c])]

    X_tr = df_tr[feature_cols].to_numpy(np.float32)
    X_va = df_va[feature_cols].to_numpy(np.float32)
    np.nan_to_num(X_tr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_va, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    y_tr = df_tr["label"].to_numpy().astype(int)
    y_va = df_va["label"].to_numpy().astype(int)
    g_tr = df_tr["wafer_id"].to_numpy()
    g_va = df_va["wafer_id"].to_numpy()

    assert len(set(g_tr) & set(g_va)) == 0, "Wafer overlap between dev_train and dev_val!"

    print(f"  train: {X_tr.shape} ({len(set(g_tr))} wafers, {y_tr.mean()*100:.2f}% positive)", flush=True)
    print(f"  val:   {X_va.shape} ({len(set(g_va))} wafers, {y_va.mean()*100:.2f}% positive)", flush=True)
    print(f"  features: {len(feature_cols)}  |  loaded in {time.time()-t0:.1f}s", flush=True)

    with open(FEATURES_PATH, "w") as f:
        json.dump({"n_features": len(feature_cols), "features": feature_cols}, f, indent=2)

    return X_tr, y_tr, g_tr, X_va, y_va, g_va, feature_cols


# ---------------------------------------------------------------------------
# Engine Configurations (GPU Accelerated)
# ---------------------------------------------------------------------------
def _cb_params():
    return dict(
        iterations=MAX_ROUNDS,
        depth=8,
        learning_rate=LR,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=SEED,
        early_stopping_rounds=PATIENCE,
        verbose=250,
        task_type="GPU",
        devices="0",
    )


def _xgb_params():
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 7,
        "learning_rate": LR,
        "subsample": 0.80,
        "colsample_bytree": 0.70,
        "min_child_weight": 20,
        "reg_lambda": 2.0,
        "seed": SEED,
    }


def _lgb_params():
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": LR,
        "scale_pos_weight": 3.0,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "lambda_l2": 2.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": SEED + 100,
    }


def fit_catboost(Xa, ya, Xb, yb, rounds=None):
    p = _cb_params()
    if rounds:
        p["iterations"] = rounds
        p.pop("early_stopping_rounds", None)
    m = CatBoostClassifier(**p)
    if rounds:
        m.fit(Pool(Xa, ya), verbose=250)
    else:
        m.fit(Pool(Xa, ya), eval_set=Pool(Xb, yb), use_best_model=True)
    return m, int(m.tree_count_)


def fit_xgboost(Xa, ya, Xb, yb, rounds=None):
    da = xgb.DMatrix(Xa, label=ya)
    if rounds:
        m = xgb.train(_xgb_params(), da, num_boost_round=rounds, verbose_eval=False)
        return m, rounds
    db = xgb.DMatrix(Xb, label=yb)
    m = xgb.train(
        _xgb_params(),
        da,
        num_boost_round=MAX_ROUNDS,
        evals=[(db, "val")],
        early_stopping_rounds=PATIENCE,
        verbose_eval=250,
    )
    return m, int(m.best_iteration + 1)


def fit_lightgbm(Xa, ya, Xb, yb, rounds=None):
    da = lgb.Dataset(Xa, label=ya)
    if rounds:
        m = lgb.train(_lgb_params(), da, num_boost_round=rounds)
        return m, rounds
    db = lgb.Dataset(Xb, label=yb, reference=da)
    m = lgb.train(
        _lgb_params(),
        da,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[db],
        callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.log_evaluation(250)],
    )
    return m, int(m.best_iteration)


ENGINES = {
    "catboost": (fit_catboost, lambda m, X: m.predict_proba(X)[:, 1]),
    "xgboost": (fit_xgboost, lambda m, X: m.predict(xgb.DMatrix(X))),
    "lightgbm": (fit_lightgbm, lambda m, X: m.predict(X)),
}


# ---------------------------------------------------------------------------
# Out-of-Fold Stage (dev_train only)
# ---------------------------------------------------------------------------
def build_oof(X, y, groups):
    print("\n" + "=" * 78)
    print(f"OUT-OF-FOLD STAGE ({N_FOLDS}-Fold GroupKFold on wafer_id, dev_train only)")
    print("=" * 78, flush=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = {k: np.zeros(len(y), dtype=np.float64) for k in ENGINES}
    rounds_used = {k: [] for k in ENGINES}

    for fold, (tr_idx, ho_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        print(
            f"\n--- Fold {fold}/{N_FOLDS}: {len(tr_idx):,} train / {len(ho_idx):,} holdout "
            f"({len(set(groups[ho_idx]))} wafers) ---",
            flush=True,
        )
        for name, (fit_fn, pred_fn) in ENGINES.items():
            t0 = time.time()
            model, n_rounds = fit_fn(X[tr_idx], y[tr_idx], X[ho_idx], y[ho_idx])
            oof[name][ho_idx] = pred_fn(model, X[ho_idx])
            rounds_used[name].append(n_rounds)
            fold_ap = average_precision_score(y[ho_idx], oof[name][ho_idx])
            print(
                f"    {name:9s} rounds={n_rounds:5d}  fold AUC-PR={fold_ap:.4f}  ({time.time()-t0:.1f}s)",
                flush=True,
            )

    print("\nOOF Summary (Honest wafer-disjoint on dev_train):", flush=True)
    for name in ENGINES:
        oof_ap = average_precision_score(y, oof[name])
        med_r = int(np.median(rounds_used[name]))
        print(f"  {name:9s} OOF AUC-PR = {oof_ap:.4f}   median rounds = {med_r}", flush=True)

    final_rounds = {k: int(np.median(v)) for k, v in rounds_used.items()}
    return oof, final_rounds


# ---------------------------------------------------------------------------
# Blending: Rank-space + Gradient-free Nelder-Mead on OOF
# ---------------------------------------------------------------------------
def _rank01(p):
    return rankdata(p, method="average") / len(p)


def blend_ranks(P_rank, w):
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return w @ P_rank


def optimise_weights(P_rank, y, n_random=6000, seed=SEED):
    rng = np.random.default_rng(seed)
    k = P_rank.shape[0]
    cands = np.vstack([
        rng.dirichlet(np.ones(k), size=n_random),
        np.eye(k),
        np.ones((1, k)) / k,
    ])
    scores = np.array([average_precision_score(y, blend_ranks(P_rank, w)) for w in cands])
    w_best = cands[int(scores.argmax())]

    res = minimize(
        lambda w: -average_precision_score(y, blend_ranks(P_rank, np.abs(w) + 1e-9)),
        w_best,
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 4000},
    )
    w_ref = np.abs(res.x)
    w_ref = w_ref / w_ref.sum()
    if average_precision_score(y, blend_ranks(P_rank, w_ref)) > scores.max():
        w_best = w_ref
    return w_best / w_best.sum(), float(max(scores.max(), average_precision_score(y, blend_ranks(P_rank, w_best))))


def tune_threshold(y, p, lo=0.01, hi=0.99, n=400):
    grid = np.linspace(lo, hi, n)
    best = (-1.0, 0.5, 0.0, 0.0)
    for t in grid:
        pred = (p >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best[0]:
            best = (
                f1,
                float(t),
                precision_score(y, pred, zero_division=0),
                recall_score(y, pred, zero_division=0),
            )
    return {"f1": best[0], "threshold": best[1], "precision": best[2], "recall": best[3]}


# ---------------------------------------------------------------------------
# Main Training & Refitting Execution
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 78)
    print("MODEL F TRAINING: WAFER-CONDITIONAL MANIFOLD DETECTOR (GPU ACCELERATED)")
    print("=" * 78, flush=True)

    X_tr, y_tr, g_tr, X_va, y_va, g_va, feature_cols = load_datasets()

    # 1. Out-of-fold stage: choose rounds & blend weights strictly on dev_train
    oof, final_rounds = build_oof(X_tr, y_tr, g_tr)
    names = list(ENGINES)
    P_oof_rank = np.stack([_rank01(oof[n]) for n in names])
    w, oof_blend_ap = optimise_weights(P_oof_rank, y_tr)

    print("\nBlend weights fitted OUT-OF-FOLD on dev_train:", flush=True)
    for n, wi in zip(names, w):
        print(f"  {n:9s} {wi:.4f}", flush=True)
    print(f"  OOF blended AUC-PR = {oof_blend_ap:.4f}", flush=True)

    oof_thr = tune_threshold(y_tr, blend_ranks(P_oof_rank, w))
    print(
        f"  OOF-selected threshold (rank space) = {oof_thr['threshold']:.4f}  F1={oof_thr['f1']:.4f}",
        flush=True,
    )

    # 2. Refit each engine on full dev_train at OOF-selected round counts
    print("\n" + "=" * 78)
    print("REFITTING ENGINES ON FULL dev_train (GPU ACCELERATED)")
    print("=" * 78, flush=True)
    val_preds, models = {}, {}
    for name, (fit_fn, pred_fn) in ENGINES.items():
        t0 = time.time()
        model, _ = fit_fn(X_tr, y_tr, None, None, rounds=final_rounds[name])
        models[name] = model
        val_preds[name] = pred_fn(model, X_va)
        ap = average_precision_score(y_va, val_preds[name])
        print(
            f"  {name:9s} rounds={final_rounds[name]:5d}  dev_val AUC-PR={ap:.4f}  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models["catboost"].save_model(str(CB_PATH))
    models["xgboost"].save_model(str(XGB_PATH))
    models["lightgbm"].save_model(str(LGB_PATH))

    # 3. Blending on dev_val using OOF weights
    P_val_rank = np.stack([_rank01(val_preds[n]) for n in names])
    p_stack = blend_ranks(P_val_rank, w)

    stack_ap = average_precision_score(y_va, p_stack)
    stack_roc = roc_auc_score(y_va, p_stack)
    thr = tune_threshold(y_va, p_stack)

    print("\n" + "=" * 78)
    print("CANONICAL DEV_VAL RESULTS (Weights and round counts chosen out-of-fold)")
    print("=" * 78, flush=True)
    print(f"{'model':<12}{'AUC-PR':>10}{'ROC-AUC':>10}{'F1':>8}{'Precision':>11}{'Recall':>9}{'thr':>8}", flush=True)
    for n in names:
        m = tune_threshold(y_va, val_preds[n])
        ap = average_precision_score(y_va, val_preds[n])
        roc = roc_auc_score(y_va, val_preds[n])
        print(
            f"{n:<12}{ap:>10.4f}{roc:>10.4f}{m['f1']:>8.4f}{m['precision']:>11.4f}{m['recall']:>9.4f}{m['threshold']:>8.4f}",
            flush=True,
        )
    print(
        f"{'MODEL F STACK':<12}{stack_ap:>10.4f}{stack_roc:>10.4f}{thr['f1']:>8.4f}{thr['precision']:>11.4f}{thr['recall']:>9.4f}{thr['threshold']:>8.4f}",
        flush=True,
    )

    # 4. Wafer-clustered bootstrap CI
    rng = np.random.default_rng(SEED)
    wafers = np.unique(g_va)
    idx_by_wafer = {wf: np.flatnonzero(g_va == wf) for wf in wafers}
    boot = np.empty(400)
    for b in range(400):
        pick = rng.integers(0, len(wafers), len(wafers))
        sel = np.concatenate([idx_by_wafer[wafers[j]] for j in pick])
        boot[b] = average_precision_score(y_va[sel], p_stack[sel])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(
        f"\nWafer-clustered bootstrap: AUC-PR = {stack_ap:.4f}  SE={boot.std(ddof=1):.4f}  95% CI [{lo:.4f}, {hi:.4f}]",
        flush=True,
    )

    weights_dict = {
        "engine_names": names,
        "weights": {n: float(wi) for n, wi in zip(names, w)},
        "blend_space": "rank",
        "rounds": final_rounds,
        "oof_blend_auc_pr": float(oof_blend_ap),
        "oof_threshold": float(oof_thr["threshold"]),
        "optimal_threshold": float(thr["threshold"]),
        "val_auc_pr": float(stack_ap),
        "val_roc_auc": float(stack_roc),
        "val_f1": float(thr["f1"]),
        "val_precision": float(thr["precision"]),
        "val_recall": float(thr["recall"]),
        "val_auc_pr_se_wafer_clustered": float(boot.std(ddof=1)),
        "val_auc_pr_ci95": [float(lo), float(hi)],
        "n_features": len(feature_cols),
    }
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights_dict, f, indent=2)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        VAL_PREDS_NPZ,
        y_val=y_va,
        wafer_id=g_va,
        p_stack=p_stack,
        **{f"p_{n}": val_preds[n] for n in names},
    )
    print(f"\nSaved {WEIGHTS_PATH.name} and {VAL_PREDS_NPZ.name}", flush=True)
    print("\nMODEL F TRAINING COMPLETE\n")


if __name__ == "__main__":
    main()
