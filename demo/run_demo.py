#!/usr/bin/env python3
"""Train SPARC on the compact scMultiSim demo and export global/dynamic GRNs."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparc-demo")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scvi
import torch

from sparc import SPARC


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(demo_dir / "data"))
    parser.add_argument("--output-dir", default=str(demo_dir / "results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def reset_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scvi.settings.seed = seed
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dense_float32(values) -> np.ndarray:
    if sp.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def gene_mask(var_names: Iterable[str], selected: Iterable[str]) -> np.ndarray:
    selected_set = set(map(str, selected))
    return np.asarray([str(gene) in selected_set for gene in var_names], dtype=bool)


def build_cascade_masks(gold: np.lib.npyio.NpzFile, n_genes: int) -> tuple[np.ndarray, np.ndarray]:
    all_genes = list(map(str, gold["gene_names"]))
    ligand_genes = list(map(str, gold["ligand_genes"]))
    receptor_genes = list(map(str, gold["receptor_genes"]))
    tf_genes = list(map(str, gold["tf_genes"]))
    gene_to_index = {gene: index for index, gene in enumerate(all_genes)}

    lr_mask = np.zeros((n_genes, n_genes), dtype=np.float32)
    for ligand_index, ligand in enumerate(ligand_genes):
        for receptor_index, receptor in enumerate(receptor_genes):
            if gold["lr_prior"][ligand_index, receptor_index]:
                lr_mask[gene_to_index[ligand], gene_to_index[receptor]] = 1.0

    receptor_tf_mask = np.zeros((n_genes, len(tf_genes)), dtype=np.float32)
    for receptor_index, receptor in enumerate(receptor_genes):
        for tf_index in range(len(tf_genes)):
            if gold["rtf_prior"][receptor_index, tf_index]:
                receptor_tf_mask[gene_to_index[receptor], tf_index] = 1.0
    return lr_mask, receptor_tf_mask


def standardized_expression(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x.astype(np.float32, copy=True)
    centered -= centered.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, ddof=1, keepdims=True)
    valid = std.ravel() > 1e-8
    standardized = np.divide(centered, std, out=np.zeros_like(centered), where=std > 1e-8)
    return standardized, valid


def build_data_soft_grn_masks(
    x: np.ndarray,
    tf_mask: np.ndarray,
    target_mask: np.ndarray,
    prior_tf_target: np.ndarray,
    corr_min: float = 0.35,
    top_per_target: int = 2,
    max_new_edges: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Construct the default SPARC data-soft TF-target support."""
    prior_tf_target = prior_tf_target.astype(bool, copy=False)
    prior_target_tf = prior_tf_target.T.astype(np.float32)
    standardized, valid = standardized_expression(x)
    x_tf = standardized[:, tf_mask]
    x_target = standardized[:, target_mask]
    corr = (x_tf.T @ x_target) / max(1, x.shape[0] - 1)
    corr[~valid[tf_mask], :] = 0.0
    corr[:, ~valid[target_mask]] = 0.0
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(corr)
    eligible = (~prior_tf_target) & (abs_corr >= corr_min)

    rows: list[dict[str, float | int]] = []
    for target_index in range(prior_tf_target.shape[1]):
        tf_indices = np.flatnonzero(eligible[:, target_index])
        order = tf_indices[np.argsort(abs_corr[tf_indices, target_index])[::-1]]
        for target_rank, tf_index in enumerate(order[:top_per_target], start=1):
            rows.append(
                {
                    "tf_index": int(tf_index),
                    "target_index": int(target_index),
                    "correlation": float(corr[tf_index, target_index]),
                    "absolute_correlation": float(abs_corr[tf_index, target_index]),
                    "target_rank": int(target_rank),
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return (
            prior_target_tf.copy(),
            prior_target_tf.copy(),
            np.zeros_like(prior_target_tf),
            candidates,
        )

    candidates = candidates.sort_values(
        ["absolute_correlation", "target_index", "tf_index"],
        ascending=[False, True, True],
    ).head(max_new_edges).reset_index(drop=True)
    candidates["global_rank"] = np.arange(1, len(candidates) + 1)
    skeleton_tf_target = prior_tf_target.copy()
    candidate_tf_target = np.zeros_like(prior_tf_target, dtype=np.float32)
    for row in candidates.itertuples(index=False):
        skeleton_tf_target[row.tf_index, row.target_index] = True
        candidate_tf_target[row.tf_index, row.target_index] = 1.0
    return (
        skeleton_tf_target.T.astype(np.float32),
        prior_target_tf,
        candidate_tf_target.T,
        candidates,
    )


def edge_table(weights: pd.DataFrame) -> pd.DataFrame:
    table = weights.rename_axis("tf").reset_index().melt(
        id_vars="tf",
        var_name="target",
        value_name="weight",
    )
    table["absolute_weight"] = table["weight"].abs()
    return table.sort_values("absolute_weight", ascending=False).reset_index(drop=True)


def dynamic_summary(
    weights: np.ndarray,
    allowed_tf_target: np.ndarray,
    tf_genes: list[str],
    target_genes: list[str],
) -> pd.DataFrame:
    rows = []
    for tf_index, target_index in np.argwhere(allowed_tf_target):
        values = weights[:, tf_index, target_index]
        rows.append(
            {
                "tf": tf_genes[tf_index],
                "target": target_genes[target_index],
                "mean_weight": float(values.mean()),
                "std_across_cells": float(values.std(ddof=1)),
                "min_weight": float(values.min()),
                "max_weight": float(values.max()),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values("std_across_cells", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    reset_seed(args.seed)
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(data_dir / "scmultisim_demo.h5ad")
    gold = np.load(data_dir / "scmultisim_demo_gold.npz", allow_pickle=True)
    x = dense_float32(adata.X)
    tf_genes = list(map(str, gold["tf_genes"]))
    target_genes = list(map(str, gold["target_genes"]))
    tf_mask = gene_mask(adata.var_names, tf_genes)
    target_mask = gene_mask(adata.var_names, target_genes)
    grn_prior = gold["grn_prior"].astype(bool)
    lr_mask, receptor_tf_mask = build_cascade_masks(gold, adata.n_vars)
    skeleton, prior_mask, candidate_mask, candidates = build_data_soft_grn_masks(
        x,
        tf_mask,
        target_mask,
        grn_prior,
    )

    run_adata = adata.copy()
    SPARC.setup_anndata(
        run_adata,
        x_layer=None,
        spatial_key="spatial",
        sigma=0.18,
        n_neighbors=12,
    )
    model = SPARC(
        run_adata,
        n_latent=12,
        n_hidden=128,
        n_layers=1,
        skeleton=skeleton,
        grn_prior_mask=prior_mask,
        grn_candidate_penalty_mask=candidate_mask,
        regulator_index=tf_mask.tolist(),
        target_index=target_mask.tolist(),
        lr_mask=lr_mask,
        rec_tf_mask=receptor_tf_mask,
        loss_preset="balanced",
        likelihood="mse",
        dropout_rate=0.05,
        dynamic_scale_init=0.25,
        tf_activity_dropout=0.0,
        lambda_omega_balance=0.0,
        omega_balance_tau=5.0,
        omega_balance_mode="penalty",
        omega_balance_iters=3,
        omega_row_max_share=0.5,
        grn_tf_load_alpha=2.5,
        lr_degree_normalize=False,
        grn_prior_penalty_weight=0.1,
        grn_candidate_penalty_weight=1.0,
    )
    model.initialize_grn_from_expression(scale=0.08)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.train(
        max_epochs=args.epochs,
        lr=1e-3,
        accelerator=args.accelerator,
        devices=args.devices,
        batch_size=args.batch_size,
        train_size=0.9,
        validation_size=0.1,
        enable_progress_bar=args.progress,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start

    # Public inference interfaces demonstrated by this example.
    global_grn = model.get_global_grn()
    dynamic_grn = model.get_dynamic_grn()

    global_grn.index.name = "tf"
    global_grn.to_csv(output_dir / "global_grn.csv")
    edge_table(global_grn).head(100).to_csv(output_dir / "top_global_edges.csv", index=False)
    allowed_tf_target = skeleton.T.astype(bool)
    dynamic_summary(dynamic_grn, allowed_tf_target, tf_genes, target_genes).to_csv(
        output_dir / "dynamic_grn_edge_summary.csv",
        index=False,
    )
    np.savez_compressed(
        output_dir / "dynamic_grn.npz",
        weights=dynamic_grn.astype(np.float32),
        cell_names=np.asarray(run_adata.obs_names, dtype=str),
        tf_genes=np.asarray(tf_genes, dtype=str),
        target_genes=np.asarray(target_genes, dtype=str),
    )

    summary = {
        "purpose": "API demonstration; not a reproduction of the full benchmark",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_cells": int(run_adata.n_obs),
        "n_genes": int(run_adata.n_vars),
        "n_tfs": len(tf_genes),
        "n_targets": len(target_genes),
        "n_prior_grn_edges": int(grn_prior.sum()),
        "n_data_soft_candidates": int(len(candidates)),
        "n_allowed_grn_edges": int(allowed_tf_target.sum()),
        "global_grn_shape": list(global_grn.shape),
        "dynamic_grn_shape": list(dynamic_grn.shape),
        "training_seconds": training_seconds,
        "device": str(next(model.module.parameters()).device),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "scvi_tools": scvi.__version__,
        "anndata": ad.__version__,
        "interfaces": ["SPARC.get_global_grn", "SPARC.get_dynamic_grn"],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
