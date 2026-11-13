#!/usr/bin/env python3
"""Train SPARC on processed Perturb-map data and run in-silico knockout."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparc-demo")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import anndata as ad
import numpy as np
import pandas as pd
import torch

from sparc import SPARC
from workflow_utils import assert_gene_alignment, load_prior_bundle, reset_seed, write_json


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(demo_dir / "perturbation" / "data" / "perturb_map_processed.h5ad"),
    )
    parser.add_argument(
        "--priors",
        default=str(demo_dir / "perturbation" / "data" / "perturb_map_priors.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(demo_dir / "perturbation" / "results"),
    )
    parser.add_argument("--genes", nargs="+", default=["Tgfbr2"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--damping", type=float, default=0.25)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def summarize_delta(delta: np.ndarray, target_genes: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "gene": target_genes,
            "predicted_mean_delta": delta.mean(axis=0),
            "predicted_mean_abs_delta": np.abs(delta).mean(axis=0),
            "predicted_l2_delta": np.sqrt(np.square(delta).mean(axis=0)),
        }
    )
    result["absolute_mean_delta"] = result["predicted_mean_delta"].abs()
    result = result.sort_values("absolute_mean_delta", ascending=False).reset_index(drop=True)
    result["absolute_response_rank"] = np.arange(1, len(result) + 1)
    return result


def main() -> None:
    args = parse_args()
    reset_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.data)
    priors = load_prior_bundle(args.priors)
    assert_gene_alignment(adata.var_names, priors)
    SPARC.setup_anndata(
        adata,
        x_layer=None,
        spatial_key="spatial",
        sigma=250.0,
        n_neighbors=6,
    )
    model = SPARC(
        adata,
        skeleton=priors["skeleton"],
        grn_prior_mask=priors["grn_prior_mask"],
        grn_candidate_penalty_mask=priors["grn_candidate_penalty_mask"],
        regulator_index=priors["regulator_index"].astype(bool).tolist(),
        target_index=priors["target_index"].astype(bool).tolist(),
        lr_mask=priors["lr_mask"],
        rec_tf_mask=priors["rec_tf_mask"],
        n_hidden=64,
        n_latent=8,
        n_layers=1,
        loss_preset="balanced",
        likelihood="mse",
        grn_prior_penalty_weight=0.1,
        grn_candidate_penalty_weight=1.0,
        tf_activity_dropout=0.0,
        omega_balance_mode="none",
        lambda_omega_balance=0.0,
        omega_balance_tau=5.0,
        omega_balance_iters=3,
        omega_row_max_share=1.0,
        grn_tf_load_alpha=0.0,
        lr_degree_normalize=True,
        lr_niche_contrast=False,
    )
    model.initialize_grn_from_expression(adata, scale=0.05)

    start = time.perf_counter()
    model.train(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        enable_progress_bar=args.progress,
        early_stopping=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    training_seconds = time.perf_counter() - start

    control_mask = adata.obs["phenotypes"].astype(str).isin(["KP_1-1", "KP_1-2", "KP_1-3"]).to_numpy()
    if not control_mask.any():
        raise ValueError("No unperturbed KP control spots were found in obs['phenotypes'].")
    completed = []
    for gene in args.genes:
        if gene not in adata.var_names:
            raise ValueError(f"Perturbation gene {gene!r} is not present in the model gene panel.")
        delta_df = model.predict_perturbation(
            adata=adata,
            gene=gene,
            ko_value=0.0,
            cell_mask=control_mask,
            n_steps=args.n_steps,
            damping=args.damping,
            return_dataframe=True,
            batch_size=args.batch_size,
        )
        target_genes = delta_df.columns.to_numpy(dtype=str)
        control_delta = delta_df.iloc[np.flatnonzero(control_mask)].to_numpy(dtype=np.float32)
        summary = summarize_delta(control_delta, target_genes)
        summary.insert(0, "perturbation_gene", gene)
        summary.to_csv(output_dir / f"prediction_{gene}.csv", index=False)
        np.savez_compressed(
            output_dir / f"delta_{gene}.npz",
            delta=control_delta.astype(np.float32),
            cell_names=np.asarray(adata.obs_names[control_mask], dtype=str),
            target_genes=target_genes.astype(str),
        )
        completed.append(gene)

    write_json(
        output_dir / "run_summary.json",
        {
            "workflow": "Perturb-map in-silico knockout",
            "data_shape": [int(adata.n_obs), int(adata.n_vars)],
            "n_control_spots": int(control_mask.sum()),
            "perturbation_genes": completed,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "rollout_steps": args.n_steps,
            "rollout_damping": args.damping,
            "training_seconds": training_seconds,
            "device": str(next(model.module.parameters()).device),
            "torch": torch.__version__,
        },
    )


if __name__ == "__main__":
    main()
