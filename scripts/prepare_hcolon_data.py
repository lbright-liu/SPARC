#!/usr/bin/env python3
"""Package the complete processed hColon training input used in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", required=True)
    parser.add_argument("--selected-cells", required=True)
    parser.add_argument("--database-dir", required=True)
    parser.add_argument("--candidate-edges", required=True)
    parser.add_argument("--output-dir", default=str(repo / "demo" / "transplantation" / "data"))
    return parser.parse_args()


def read_edges(path: Path, genes: list[str], *, remove_self_loops: bool = False) -> np.ndarray:
    table = pd.read_csv(path, usecols=["source", "target"])
    table["source"] = table["source"].astype(str).str.strip().str.upper()
    table["target"] = table["target"].astype(str).str.strip().str.upper()
    lookup = {gene.strip().upper(): index for index, gene in enumerate(genes)}
    source_index = table["source"].map(lookup)
    target_index = table["target"].map(lookup)
    valid = source_index.notna() & target_index.notna()
    mask = np.zeros((len(genes), len(genes)), dtype=np.float32)
    mask[source_index[valid].astype(int), target_index[valid].astype(int)] = 1.0
    if remove_self_loops:
        np.fill_diagonal(mask, 0.0)
    return mask


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(args.source_h5ad)
    counts = source.layers["counts"] if "counts" in source.layers else source.X
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    counts = counts.astype(np.float32)

    obs_columns = [column for column in ["cell_type", "marker_domain"] if column in source.obs]
    obs = source.obs[obs_columns].copy()
    obs["transplant_role"] = "other_training_cell"
    selected = pd.read_csv(args.selected_cells)
    selected = selected[selected["in_training"].astype(bool)]
    role_names = {
        "normal_like_fibroblast_donor": "donor",
        "local_tumor_like_environment": "recipient_environment",
    }
    for paper_role, public_role in role_names.items():
        names = selected.loc[selected["role"] == paper_role, "obs_name"].astype(str)
        matched = obs.index.intersection(names)
        obs.loc[matched, "transplant_role"] = public_role

    packaged = ad.AnnData(X=counts.copy(), obs=obs, var=source.var.copy())
    packaged.layers["counts"] = counts
    packaged.obsm["spatial"] = np.asarray(source.obsm["spatial"], dtype=np.float32)
    packaged.uns["sparc_data_provenance"] = {
        "dataset": "10x Genomics Xenium human colon cancer",
        "scope": "complete held-out SPARC training input; no cell or gene subsampling",
        "heldout_truth_cells_excluded_before_packaging": 66,
        "preprocessing_required": "counts -> normalize_total(10000) -> log1p",
    }
    packaged.write_h5ad(output_dir / "hcolon_processed_counts.h5ad", compression="gzip")

    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(
        10000.0,
        totals,
        out=np.zeros_like(totals, dtype=np.float32),
        where=totals > 0,
    )
    lognorm = counts.multiply(scale[:, None]).tocsr()
    lognorm.data = np.log1p(lognorm.data)
    environment_names = selected.loc[
        selected["role"] == "local_tumor_like_environment", "obs_name"
    ].astype(str)
    environment_indices = packaged.obs_names.get_indexer(environment_names)
    if (environment_indices < 0).any():
        raise ValueError("Some selected recipient-environment cells are absent from the training input.")
    environment_coords = np.asarray(packaged.obsm["spatial"])[environment_indices]
    target_rows = pd.read_csv(args.selected_cells)
    target_rows = target_rows[target_rows["role"] == "heldout_tumor_like_fibroblast_truth"]
    local_profiles = []
    for row in target_rows.itertuples(index=False):
        center = np.asarray([row.x, row.y], dtype=np.float32)
        distance = np.sqrt(np.square(environment_coords - center[None, :]).sum(axis=1))
        nearest = environment_indices[np.argsort(distance)[:64]]
        local_profiles.append(np.asarray(lognorm[nearest].mean(axis=0)).ravel())
    np.savez_compressed(
        output_dir / "hcolon_target_local_niches.npz",
        profiles=np.asarray(local_profiles, dtype=np.float32),
        context_names=target_rows["obs_name"].to_numpy(dtype=str),
        gene_names=np.asarray(packaged.var_names, dtype=str),
        neighbors_per_profile=np.asarray(64, dtype=np.int64),
    )

    genes = list(map(str, packaged.var_names))
    database_dir = Path(args.database_dir)
    lr_mask = read_edges(database_dir / "LigRecDB.csv", genes)
    rec_tf_full = read_edges(database_dir / "RecTFDB.csv", genes)
    strict = read_edges(database_dir / "TFTGDB.csv", genes, remove_self_loops=True)
    candidates = pd.read_csv(args.candidate_edges)
    candidate_mask = np.zeros_like(strict, dtype=np.float32)
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    for row in candidates.itertuples(index=False):
        tf_index = gene_to_index.get(str(row.tf))
        target_index = gene_to_index.get(str(row.target))
        if tf_index is not None and target_index is not None:
            candidate_mask[tf_index, target_index] = 1.0
    skeleton = np.maximum(strict, candidate_mask)
    regulator_index = skeleton.sum(axis=1) > 0
    target_index = np.ones(len(genes), dtype=bool)

    np.savez_compressed(
        output_dir / "hcolon_priors.npz",
        gene_names=np.asarray(genes, dtype=str),
        skeleton=skeleton,
        grn_prior_mask=strict,
        grn_candidate_penalty_mask=candidate_mask,
        regulator_index=regulator_index,
        target_index=target_index,
        lr_mask=lr_mask,
        rec_tf_mask=rec_tf_full[:, regulator_index],
    )
    metadata = {
        "n_training_cells": int(packaged.n_obs),
        "n_genes": int(packaged.n_vars),
        "n_donor_cells": int((obs["transplant_role"] == "donor").sum()),
        "n_recipient_environment_cells": int(
            (obs["transplant_role"] == "recipient_environment").sum()
        ),
        "n_heldout_truth_cells": 66,
        "n_target_local_niche_profiles": int(len(local_profiles)),
        "target_niche_neighbors": 64,
        "n_lr_edges": int(lr_mask.sum()),
        "n_receptor_tf_edges": int(rec_tf_full.sum()),
        "n_strict_grn_edges": int(strict.sum()),
        "n_data_soft_edges": int(candidate_mask.sum()),
        "n_allowed_grn_edges": int(skeleton.sum()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
