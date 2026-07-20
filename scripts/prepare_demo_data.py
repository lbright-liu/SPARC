#!/usr/bin/env python3
"""Create the compact, domain-balanced scMultiSim dataset shipped with SPARC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing adata_benchmark.h5ad and grn_gold_standard.npz.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "demo" / "data"),
    )
    parser.add_argument("--cells-per-domain", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def dense_float32(values) -> np.ndarray:
    if sp.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(source_dir / "adata_benchmark.h5ad")
    gold = np.load(source_dir / "grn_gold_standard.npz", allow_pickle=True)
    if "domain" not in adata.obs:
        raise KeyError("The source AnnData must contain obs['domain'].")

    rng = np.random.default_rng(args.seed)
    selected: list[int] = []
    domains = adata.obs["domain"].astype(str).to_numpy()
    for domain in sorted(pd.unique(domains)):
        candidates = np.flatnonzero(domains == domain)
        if candidates.size < args.cells_per_domain:
            raise ValueError(
                f"Domain {domain!r} contains {candidates.size} cells; "
                f"cannot sample {args.cells_per_domain}."
            )
        selected.extend(rng.choice(candidates, size=args.cells_per_domain, replace=False).tolist())
    selected_array = np.sort(np.asarray(selected, dtype=np.int64))

    source_x = adata.layers["lognorm"] if "lognorm" in adata.layers else adata.X
    x = dense_float32(source_x[selected_array])
    obs = adata.obs.iloc[selected_array][["domain"]].copy()
    obs.index = pd.Index([f"DemoCell_{i + 1}" for i in range(len(selected_array))])
    demo = ad.AnnData(X=x, obs=obs, var=adata.var.copy())
    demo.obsm["spatial"] = np.asarray(adata.obsm["spatial"])[selected_array].astype(np.float32)
    demo.uns["sparc_demo"] = {
        "source_dataset": "data_official_sparc_balanced_n3000_g300",
        "source_seed": 17,
        "sampling_seed": args.seed,
        "cells_per_domain": args.cells_per_domain,
        "expression_scale": "library-size normalized and log1p transformed",
    }
    demo.write_h5ad(output_dir / "scmultisim_demo.h5ad", compression="gzip")

    arrays: dict[str, np.ndarray] = {}
    for key in gold.files:
        if key == "W_dynamic":
            arrays[key] = gold[key][selected_array].astype(np.float32)
        elif key == "domain":
            arrays[key] = gold[key][selected_array]
        else:
            arrays[key] = gold[key]
    np.savez_compressed(output_dir / "scmultisim_demo_gold.npz", **arrays)

    metadata = {
        "source_dataset": "scMultiSim-v5 high-noise-prior benchmark",
        "source_directory_name": source_dir.name,
        "sampling_seed": args.seed,
        "cells_per_domain": args.cells_per_domain,
        "n_cells": int(demo.n_obs),
        "n_genes": int(demo.n_vars),
        "domains": obs["domain"].value_counts().sort_index().to_dict(),
        "n_ligands": int(len(gold["ligand_genes"])),
        "n_receptors": int(len(gold["receptor_genes"])),
        "n_tfs": int(len(gold["tf_genes"])),
        "n_targets": int(len(gold["target_genes"])),
        "n_grn_prior_edges": int(gold["grn_prior"].sum()),
        "n_true_grn_edges": int(gold["grn_true"].sum()),
        "n_true_dynamic_edges": int(gold["dynamic_mask"].sum()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
