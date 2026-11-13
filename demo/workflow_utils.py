"""Shared utilities for the executable SPARC workflow examples."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import scvi
import torch


def reset_seed(seed: int) -> None:
    """Seed the libraries used by SPARC and request deterministic kernels."""
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


def load_prior_bundle(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a gene-aligned cascade-prior bundle."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as bundle:
        required = {
            "gene_names",
            "skeleton",
            "grn_prior_mask",
            "grn_candidate_penalty_mask",
            "regulator_index",
            "target_index",
            "lr_mask",
            "rec_tf_mask",
        }
        missing = required.difference(bundle.files)
        if missing:
            raise KeyError(f"Prior bundle {path} is missing: {sorted(missing)}")
        priors = {key: bundle[key] for key in required}

    n_genes = len(priors["gene_names"])
    square_keys = ["skeleton", "grn_prior_mask", "grn_candidate_penalty_mask", "lr_mask"]
    for key in square_keys:
        if priors[key].shape != (n_genes, n_genes):
            raise ValueError(f"{key} has shape {priors[key].shape}; expected {(n_genes, n_genes)}")
    if priors["rec_tf_mask"].shape[0] != n_genes:
        raise ValueError("rec_tf_mask is not aligned to the bundled genes.")
    return priors


def assert_gene_alignment(var_names, priors: dict[str, np.ndarray]) -> None:
    observed = np.asarray(var_names, dtype=str)
    expected = priors["gene_names"].astype(str)
    if not np.array_equal(observed, expected):
        raise ValueError(
            "AnnData genes and prior bundle are not in the same order. "
            "Rebuild the processed input with the matching preparation script."
        )


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
