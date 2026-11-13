#!/usr/bin/env python3
"""Package the complete processed Perturb-map model input used in the paper."""

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
    parser.add_argument("--candidate-edges", required=True)
    parser.add_argument("--output-dir", default=str(repo / "demo" / "perturbation" / "data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(args.source_h5ad)
    expression = source.layers["lognorm"] if "lognorm" in source.layers else source.X
    expression = expression.tocsr() if sp.issparse(expression) else sp.csr_matrix(expression)
    expression = expression.astype(np.float32)

    keep_obs = [column for column in ["phenotypes", "auto_lesion"] if column in source.obs]
    keep_var = [column for column in ["gene_ids"] if column in source.var]
    packaged = ad.AnnData(
        X=expression,
        obs=source.obs[keep_obs].copy(),
        var=source.var[keep_var].copy(),
    )
    packaged.obsm["spatial"] = np.asarray(source.obsm["spatial"], dtype=np.float32)
    packaged.uns["sparc_data_provenance"] = {
        "dataset": "Perturb-map KP_1",
        "accession": "GSM5808054",
        "scope": "complete SPARC non-KO tissue model input; no cell or gene subsampling",
        "expression": "library-size normalized to 10000 and log1p transformed",
        "paper_training_cells": int(source.n_obs),
        "paper_gene_panel": int(source.n_vars),
    }
    packaged.write_h5ad(output_dir / "perturb_map_processed.h5ad", compression="gzip")

    genes = np.asarray(source.var_names, dtype=str)
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    strict = np.asarray(source.uns["skeleton"], dtype=np.float32)
    candidates = pd.read_csv(args.candidate_edges)
    candidate_mask = np.zeros_like(strict, dtype=np.float32)
    for row in candidates.itertuples(index=False):
        tf_index = gene_to_index.get(str(row.tf))
        target_index = gene_to_index.get(str(row.target))
        if tf_index is not None and target_index is not None:
            candidate_mask[tf_index, target_index] = 1.0
    skeleton = np.maximum(strict, candidate_mask)
    regulator_index = skeleton.sum(axis=1) > 0
    target_index = np.asarray(source.uns["target_index"], dtype=bool)

    np.savez_compressed(
        output_dir / "perturb_map_priors.npz",
        gene_names=genes,
        skeleton=skeleton,
        grn_prior_mask=strict,
        grn_candidate_penalty_mask=candidate_mask,
        regulator_index=regulator_index,
        target_index=target_index,
        lr_mask=np.asarray(source.uns["lr_mask"], dtype=np.float32),
        rec_tf_mask=np.asarray(source.uns["rec_tf_mask"], dtype=np.float32)[:, regulator_index],
    )
    metadata = {
        "n_cells": int(packaged.n_obs),
        "n_genes": int(packaged.n_vars),
        "n_control_spots": int(
            packaged.obs["phenotypes"].astype(str).isin(["KP_1-1", "KP_1-2", "KP_1-3"]).sum()
        ),
        "n_lr_edges": int(np.asarray(source.uns["lr_mask"]).sum()),
        "n_receptor_tf_edges": int(np.asarray(source.uns["rec_tf_mask"]).sum()),
        "n_strict_grn_edges": int(strict.sum()),
        "n_data_soft_edges": int(candidate_mask.sum()),
        "n_allowed_grn_edges": int(skeleton.sum()),
        "perturbation_genes": [gene for gene in ["Tgfbr2", "Jak2"] if gene in gene_to_index],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
