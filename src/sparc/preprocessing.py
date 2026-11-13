"""Utilities for compiling SPARC pathway priors for a gene panel."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)


def _gene_key(gene: str) -> str:
    return str(gene).strip().upper()


def _read_edge_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table.rename(columns={column: str(column).lstrip("\ufeff").strip() for column in table})
    if "source" not in table or "target" not in table:
        raise ValueError(f"{path} must contain source and target columns.")
    table["source"] = table["source"].astype(str).str.strip()
    table["target"] = table["target"].astype(str).str.strip()
    table = table[(table["source"] != "") & (table["target"] != "")]
    return table.dropna(subset=["source", "target"])


def _filter_top_edges(
    edges: pd.DataFrame,
    *,
    group_col: str,
    top_k: Optional[int],
    min_score: Optional[float],
) -> pd.DataFrame:
    if "score" not in edges:
        return edges
    edges = edges.copy()
    edges["score"] = pd.to_numeric(edges["score"], errors="coerce").fillna(1.0)
    if min_score is not None:
        edges = edges[edges["score"] >= min_score]
    if top_k is not None and top_k > 0:
        edges = (
            edges.sort_values("score", ascending=False)
            .groupby(group_col, as_index=False, group_keys=False)
            .head(top_k)
        )
    return edges.drop_duplicates(["source", "target"])


def _edge_mask(
    edges: pd.DataFrame,
    gene_to_index: dict[str, int],
    n_genes: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    source_index = edges["source"].map(lambda gene: gene_to_index.get(_gene_key(gene)))
    target_index = edges["target"].map(lambda gene: gene_to_index.get(_gene_key(gene)))
    valid = source_index.notna() & target_index.notna()
    selected = edges.loc[valid].drop_duplicates(["source", "target"]).copy()
    mask = np.zeros((n_genes, n_genes), dtype=np.float32)
    mask[source_index[valid].astype(int), target_index[valid].astype(int)] = 1.0
    return mask, selected


def build_cascade_priors(
    adata: AnnData,
    database_dir: str | Path,
    *,
    top_rec_tfs_per_receptor: Optional[int] = 0,
    top_targets_per_tf: Optional[int] = 0,
    rec_tf_min_score: Optional[float] = None,
    tf_tg_min_score: Optional[float] = None,
    allow_tf_self_loops: bool = False,
    copy: bool = False,
) -> dict:
    """Compile LR, receptor-TF, and TF-target tables into aligned masks.

    ``database_dir`` must contain ``LigRecDB.csv``, ``RecTFDB.csv``, and
    ``TFTGDB.csv`` with ``source`` and ``target`` columns. The returned masks
    follow ``adata.var_names`` exactly and can be passed directly to
    :class:`sparc.SPARC`.
    """
    if copy:
        adata = adata.copy()
    database_dir = Path(database_dir)
    lr_edges = _read_edge_table(database_dir / "LigRecDB.csv")
    rec_tf_edges = _read_edge_table(database_dir / "RecTFDB.csv")
    tf_tg_edges = _read_edge_table(database_dir / "TFTGDB.csv")

    rec_tf_edges = _filter_top_edges(
        rec_tf_edges,
        group_col="source",
        top_k=top_rec_tfs_per_receptor,
        min_score=rec_tf_min_score,
    )
    tf_tg_edges = _filter_top_edges(
        tf_tg_edges,
        group_col="source",
        top_k=top_targets_per_tf,
        min_score=tf_tg_min_score,
    )
    if not allow_tf_self_loops:
        tf_tg_edges = tf_tg_edges[
            tf_tg_edges["source"].map(_gene_key) != tf_tg_edges["target"].map(_gene_key)
        ]

    gene_to_index = {_gene_key(gene): index for index, gene in enumerate(adata.var_names)}
    lr_mask, lr_valid = _edge_mask(lr_edges, gene_to_index, adata.n_vars)
    rec_tf_mask, rec_tf_valid = _edge_mask(rec_tf_edges, gene_to_index, adata.n_vars)
    skeleton, tf_tg_valid = _edge_mask(tf_tg_edges, gene_to_index, adata.n_vars)
    if not allow_tf_self_loops:
        np.fill_diagonal(skeleton, 0.0)

    regulator_index = skeleton.sum(axis=1) > 0
    if not regulator_index.any():
        raise ValueError("No TFTGDB regulator genes overlap adata.var_names.")
    target_index = np.ones(adata.n_vars, dtype=bool)
    stats = {
        "n_genes": int(adata.n_vars),
        "n_lr_edges": int(lr_mask.sum()),
        "n_rec_tf_edges": int(rec_tf_mask.sum()),
        "n_tf_tg_edges": int(skeleton.sum()),
        "n_regulators": int(regulator_index.sum()),
        "n_targets": int(target_index.sum()),
        "top_rec_tfs_per_receptor": top_rec_tfs_per_receptor,
        "top_targets_per_tf": top_targets_per_tf,
        "rec_tf_min_score": rec_tf_min_score,
        "tf_tg_min_score": tf_tg_min_score,
        "allow_tf_self_loops": allow_tf_self_loops,
    }

    adata.uns["sparc_cascade_prior_stats"] = stats
    result = {
        "skeleton": skeleton,
        "lr_mask": lr_mask,
        "rec_tf_mask": rec_tf_mask,
        "regulator_index": regulator_index.tolist(),
        "target_index": target_index.tolist(),
        "valid_edges": {"lr": lr_valid, "rec_tf": rec_tf_valid, "tf_tg": tf_tg_valid},
        "stats": stats,
        "adata": adata if copy else None,
    }
    logger.info(
        "Cascade prior compiled: LR=%d, RecTF=%d, TFTG=%d",
        stats["n_lr_edges"],
        stats["n_rec_tf_edges"],
        stats["n_tf_tg_edges"],
    )
    return result
