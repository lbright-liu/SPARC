#!/usr/bin/env python3
"""Train SPARC on hColon and run virtual microenvironment transplantation."""

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
import scipy.sparse as sp
import torch

from sparc import SPARC
from workflow_utils import assert_gene_alignment, load_prior_bundle, reset_seed, write_json


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(demo_dir / "transplantation" / "data" / "hcolon_processed_counts.h5ad"),
    )
    parser.add_argument(
        "--priors",
        default=str(demo_dir / "transplantation" / "data" / "hcolon_priors.npz"),
    )
    parser.add_argument(
        "--local-niches",
        default=str(
            demo_dir / "transplantation" / "data" / "hcolon_target_local_niches.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(demo_dir / "transplantation" / "results"),
    )
    parser.add_argument("--normalize-total", type=float, default=10000.0)
    parser.add_argument(
        "--scenario",
        choices=["target_local", "mean_environment"],
        default="target_local",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def normalize_counts(adata: ad.AnnData, target_sum: float) -> None:
    """Reproduce the hColon counts-to-log-expression preprocessing."""
    if "counts" not in adata.layers:
        raise KeyError("hColon input must contain raw counts in adata.layers['counts'].")
    counts = adata.layers["counts"]
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    counts = counts.astype(np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(
        float(target_sum),
        totals,
        out=np.zeros_like(totals, dtype=np.float32),
        where=totals > 0,
    )
    lognorm = counts.multiply(scale[:, None]).tocsr()
    lognorm.data = np.log1p(lognorm.data)
    adata.X = lognorm
    adata.layers["lognorm"] = lognorm.copy()


def main() -> None:
    args = parse_args()
    reset_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.data)
    normalize_counts(adata, args.normalize_total)
    priors = load_prior_bundle(args.priors)
    assert_gene_alignment(adata.var_names, priors)
    SPARC.setup_anndata(
        adata,
        x_layer="lognorm",
        spatial_key="spatial",
        sigma=80.0,
        n_neighbors=15,
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
        n_hidden=96,
        n_latent=12,
        n_layers=1,
        loss_preset="balanced",
        likelihood="mse",
        grn_prior_penalty_weight=0.1,
        grn_candidate_penalty_weight=1.0,
    )
    model.initialize_grn_from_expression(adata, layer="lognorm", scale=0.05)

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

    roles = adata.obs["transplant_role"].astype(str)
    donor_mask = (roles == "donor").to_numpy()
    environment_mask = (roles == "recipient_environment").to_numpy()
    context_names = None
    if args.scenario == "target_local":
        with np.load(args.local_niches, allow_pickle=False) as local_niches:
            if not np.array_equal(
                local_niches["gene_names"].astype(str), np.asarray(adata.var_names, dtype=str)
            ):
                raise ValueError("Local niche profiles are not aligned to adata.var_names.")
            profiles = local_niches["profiles"].astype(np.float32)
            context_names = local_niches["context_names"].astype(str)
        result = model.predict_microenvironment_transplant(
            adata=adata,
            donor_mask=donor_mask,
            x_niche=profiles,
            cartesian_niche=True,
            batch_size=args.batch_size,
            return_dataframe=True,
        )
    else:
        result = model.predict_microenvironment_transplant(
            adata=adata,
            donor_mask=donor_mask,
            environment_mask=environment_mask,
            batch_size=args.batch_size,
            return_dataframe=True,
        )
    result["baseline_df"].to_csv(output_dir / "donor_baseline.csv")
    result["transplanted_df"].to_csv(output_dir / "donor_transplanted.csv")
    result["delta_df"].to_csv(output_dir / "donor_transplant_delta.csv")

    gene_summary = pd.DataFrame(
        {
            "gene": result["target_genes"],
            "baseline_mean": result["baseline"].mean(axis=0),
            "transplanted_mean": result["transplanted"].mean(axis=0),
            "predicted_mean_delta": result["delta"].mean(axis=0),
            "predicted_mean_abs_delta": np.abs(result["delta"]).mean(axis=0),
        }
    )
    gene_summary["absolute_mean_delta"] = gene_summary["predicted_mean_delta"].abs()
    gene_summary = gene_summary.sort_values("absolute_mean_delta", ascending=False)
    gene_summary.to_csv(output_dir / "transplant_gene_summary.csv", index=False)
    np.savez_compressed(
        output_dir / "transplant_predictions.npz",
        baseline=result["baseline"].astype(np.float32),
        transplanted=result["transplanted"].astype(np.float32),
        delta=result["delta"].astype(np.float32),
        donor_names=np.asarray(adata.obs_names[result["donor_index"]], dtype=str),
        target_genes=np.asarray(result["target_genes"], dtype=str),
        environment_profile_index=np.asarray(
            result.get("environment_profile_index", []), dtype=np.int64
        ),
        context_names=np.asarray(context_names if context_names is not None else [], dtype=str),
    )
    write_json(
        output_dir / "run_summary.json",
        {
            "workflow": "hColon virtual microenvironment transplantation",
            "scenario": args.scenario,
            "data_shape": [int(adata.n_obs), int(adata.n_vars)],
            "n_donor_cells": int(donor_mask.sum()),
            "n_recipient_environment_cells": int(environment_mask.sum()),
            "n_target_local_niche_profiles": int(0 if context_names is None else len(context_names)),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "normalize_total": args.normalize_total,
            "training_seconds": training_seconds,
            "device": str(next(model.module.parameters()).device),
            "torch": torch.__version__,
        },
    )


if __name__ == "__main__":
    main()
