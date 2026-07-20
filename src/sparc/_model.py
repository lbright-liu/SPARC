"""High-level SPARC model class for training and inference.

Built on top of scvi-tools BaseModelClass.
v4 changes:
  - Single-latent pathway-constrained spatial GRN architecture.
  - Spatial niche information enters only through LR -> RTF -> TF activity.
  - L-R-TF-TG cascade priors constrain LR, RecTF, and TF-TG paths.
  - Perturbation rollout is aligned with the training forward graph.
  - Virtual microenvironment transplantation keeps receiver cells fixed while
    replacing only their prior-constrained niche input.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData
from sklearn.neighbors import kneighbors_graph
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField, ObsmField
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.utils import setup_anndata_dsp

from ._constants import REGISTRY_KEYS
from ._module import SPARCVAE

logger = logging.getLogger(__name__)


LOSS_PRESETS = {
    "strict_cascade": {
        "beta": 1.0,
        "lambda_struct": 3e-4,
        "lambda_dyn": 3e-6,
        "normalized_regularization": True,
    },
    "balanced": {
        "beta": 1.0,
        "lambda_struct": 1e-4,
        "lambda_dyn": 1e-6,
        "normalized_regularization": True,
    },
    "exploratory": {
        "beta": 0.5,
        "lambda_struct": 3e-5,
        "lambda_dyn": 5e-7,
        "normalized_regularization": True,
    },
}


class SPARC(UnsupervisedTrainingMixin, BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 20,
        n_hidden: int = 256,
        n_layers: int = 1,
        skeleton: np.ndarray = None,
        grn_prior_mask: np.ndarray = None,
        regulator_index: list = None,
        target_index: list = None,
        regulator_list: list = None,
        lr_mask: np.ndarray = None,
        rec_tf_mask: np.ndarray = None,
        pathway_tf_mask: np.ndarray = None,
        encoder_input_mask: np.ndarray = None,
        disable_dynamic_grn: bool = False,
        fixed_lr_weights: np.ndarray = None,
        fixed_omega_weights: np.ndarray = None,
        grn_candidate_penalty_mask: np.ndarray = None,

        lam: float = 1.0,
        lam4_lr: float = 1.0,
        lam_dyn: float = 1e-4,
        lam_omega: float = 1e-4,

        loss_preset: Optional[str] = None,
        beta: float = 1.0,
        lambda_struct: Optional[float] = None,
        lambda_dyn: Optional[float] = None,

        normalized_regularization: Optional[bool] = None,
        **module_kwargs,
    ):
        super().__init__(adata)

        if loss_preset is not None:
            if loss_preset not in LOSS_PRESETS:
                valid = ", ".join(sorted(LOSS_PRESETS))
                raise ValueError(f"Unknown loss_preset '{loss_preset}'. Valid presets: {valid}.")
            preset = LOSS_PRESETS[loss_preset]
            beta = preset["beta"]
            lambda_struct = preset["lambda_struct"]
            lam_dyn = preset["lambda_dyn"]
            if normalized_regularization is None:
                normalized_regularization = preset["normalized_regularization"]

        if lambda_struct is not None:
            lambda_struct = float(lambda_struct)
        else:
            lambda_struct = float(lam)
        if lambda_dyn is not None:
            lam_dyn = float(lambda_dyn)
        if normalized_regularization is None:
            normalized_regularization = True

        n_genes = adata.n_vars

        if regulator_index is None:
            regulator_index = [True] * n_genes
        if target_index is None:
            target_index = [True] * n_genes
        if skeleton is None:
            skeleton = np.ones((sum(target_index), sum(regulator_index)))

        expected_shape = (sum(target_index), sum(regulator_index))
        reg_names = np.asarray(adata.var_names)[regulator_index]
        target_names = np.asarray(adata.var_names)[target_index]
        self_loop_mask = torch.tensor(
            target_names[:, None] == reg_names[None, :],
            dtype=torch.bool,
        )

        def _prepare_grn_mask(mask, name: str) -> torch.Tensor:
            if isinstance(mask, pd.DataFrame):
                mask = mask.values
            out = torch.tensor(mask, dtype=torch.float32)
            if out.shape[0] == n_genes and out.shape[1] == n_genes:
                out = out[regulator_index, :][:, target_index]
            if out.shape == (sum(regulator_index), sum(target_index)):
                out = out.T
            if out.shape != expected_shape:
                raise ValueError(
                    f"Expected {name} shape {expected_shape}, got {tuple(out.shape)}"
                )
            if self_loop_mask.any():
                out = out.clone()
                out[self_loop_mask] = 0.0
            return out

        m_grn = _prepare_grn_mask(skeleton, "skeleton")
        if grn_prior_mask is None:
            m_grn_prior = m_grn.clone()
        else:
            m_grn_prior = _prepare_grn_mask(grn_prior_mask, "grn_prior_mask")
            m_grn_prior = ((m_grn_prior > 0) & (m_grn > 0)).to(dtype=torch.float32)
        if grn_candidate_penalty_mask is None:
            m_grn_candidate_penalty = None
        else:
            m_grn_candidate_penalty = _prepare_grn_mask(
                grn_candidate_penalty_mask,
                "grn_candidate_penalty_mask",
            )
            m_grn_candidate_penalty = torch.clamp(m_grn_candidate_penalty, min=0.0)

        if lr_mask is not None:
            lr_mask = np.asarray(lr_mask, dtype=np.float32)
            if lr_mask.shape != (n_genes, n_genes):
                raise ValueError(
                    f"Expected lr_mask shape {(n_genes, n_genes)}, got {lr_mask.shape}"
                )
        if rec_tf_mask is not None:
            rec_tf_mask = np.asarray(rec_tf_mask, dtype=np.float32)
            valid_shapes = {(n_genes, n_genes), (n_genes, sum(regulator_index))}
            if rec_tf_mask.shape not in valid_shapes:
                raise ValueError(
                    "Expected rec_tf_mask shape "
                    f"{(n_genes, n_genes)} or {(n_genes, sum(regulator_index))}, "
                    f"got {rec_tf_mask.shape}"
                )
        if pathway_tf_mask is not None:
            pathway_tf_mask = np.asarray(pathway_tf_mask, dtype=np.float32)
            valid_shapes = {(n_genes, n_genes), (n_genes, sum(regulator_index))}
            if pathway_tf_mask.shape not in valid_shapes:
                raise ValueError(
                    "Expected pathway_tf_mask shape "
                    f"{(n_genes, n_genes)} or {(n_genes, sum(regulator_index))}, "
                    f"got {pathway_tf_mask.shape}"
                )
        if encoder_input_mask is not None:
            encoder_input_mask = np.asarray(encoder_input_mask, dtype=np.float32).reshape(-1)
            if encoder_input_mask.shape[0] != n_genes:
                raise ValueError(
                    f"Expected encoder_input_mask length {n_genes}, "
                    f"got {encoder_input_mask.shape[0]}"
                )
        self.module = SPARCVAE(
            n_input=n_genes,
            regulator_index=regulator_index,
            target_index=target_index,
            skeleton=m_grn,
            grn_prior_mask=m_grn_prior,
            lr_mask=lr_mask,
            rec_tf_mask=rec_tf_mask,
            pathway_tf_mask=pathway_tf_mask,
            encoder_input_mask=encoder_input_mask,
            disable_dynamic_grn=disable_dynamic_grn,
            fixed_lr_weights=fixed_lr_weights,
            fixed_omega_weights=fixed_omega_weights,
            grn_candidate_penalty_mask=m_grn_candidate_penalty,

            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            lambda_struct=lambda_struct,
            beta=beta,
            lam=lam,
            lam2=lam4_lr,
            lam_dyn=lam_dyn,
            lam_omega=lam_omega,

            normalized_regularization=normalized_regularization,
            **module_kwargs,
        )

        self._model_summary_string = (
            "SPARC model with params:\n"
            f"  n_latent: {n_latent}, n_hidden: {n_hidden}\n"
            f"  loss_preset: {loss_preset or 'custom'}\n"
            f"  beta: {beta}, lambda_struct: {lambda_struct}, "
            f"lambda_dyn: {lam_dyn}"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        x_layer: str = None,
        spatial_key: str = None,
        sigma: float = 50.0,
        n_neighbors: int = 15,
        **kwargs,
    ):
        """Register expression and spatial data for SPARC.

        The registered expression matrix is expected to be log-normalized
        expression from the upstream spatial transcriptomics preprocessing
        workflow.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [LayerField(REGISTRY_KEYS.X_KEY, x_layer, is_count_data=False)]
        adata.uns["_sparc_x_layer"] = x_layer if x_layer is not None else ""

        if spatial_key is not None:
            anndata_fields.append(ObsmField(REGISTRY_KEYS.SPATIAL_KEY, spatial_key))
            anndata_fields.append(ObsmField(REGISTRY_KEYS.X_NICHE_KEY, "X_niche"))
            spatial_coords = adata.obsm[spatial_key]
            a = kneighbors_graph(
                spatial_coords,
                n_neighbors,
                mode="distance",
                include_self=False,
            )
            a.data = np.exp(-(a.data ** 2) / (2 * sigma ** 2))
            row_sums = np.array(a.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            a = sp.diags(1.0 / row_sums) @ a
            adata.obsp["spatial_connectivities"] = a.tocsr()

            x_raw = adata.layers[x_layer] if x_layer is not None else adata.X
            x_dense = x_raw.toarray() if sp.issparse(x_raw) else np.asarray(x_raw)
            adata.obsm["X_niche"] = a @ x_dense.astype(np.float32, copy=False)

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @staticmethod
    def _get_registered_x(adata: AnnData):
        x_layer = adata.uns.get("_sparc_x_layer", "")
        if x_layer:
            if x_layer not in adata.layers:
                raise KeyError(f"Registered x_layer '{x_layer}' is not present in adata.layers.")
            return adata.layers[x_layer]
        return adata.X

    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-3,
        accelerator: str = "auto",
        devices: str = "auto",
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = None,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        if batch_size is None:
            batch_size = 256

        if plan_kwargs is None:
            plan_kwargs = {}
        plan_kwargs.setdefault("lr", lr)

        super().train(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            plan_kwargs=plan_kwargs,
            **trainer_kwargs,
        )

    @torch.no_grad()
    def initialize_grn_from_expression(
        self,
        adata: AnnData = None,
        layer: str = None,
        scale: float = 0.05,
        method: str = "corr",
    ) -> None:
        """Initialize the global GRN backbone from expression association.

        This is a warm start for dense discovery settings. It does not change
        the skeleton; off-mask weights remain zero.
        """
        if method != "corr":
            raise ValueError("Only method='corr' is currently supported.")
        adata = self._validate_anndata(adata)
        x_raw = adata.layers[layer] if layer is not None else self._get_registered_x(adata)
        x = x_raw.toarray() if sp.issparse(x_raw) else np.asarray(x_raw)
        x = x.astype(np.float32, copy=False)

        reg_mask = np.asarray(self.module.regulator_index, dtype=bool)
        target_mask = np.asarray(self.module.target_index, dtype=bool)
        x_reg = x[:, reg_mask]
        x_target = x[:, target_mask]
        x_reg = x_reg - x_reg.mean(axis=0, keepdims=True)
        x_target = x_target - x_target.mean(axis=0, keepdims=True)

        denom = np.sqrt((x_reg**2).sum(axis=0, keepdims=True)).T @ np.sqrt(
            (x_target**2).sum(axis=0, keepdims=True)
        )
        denom[denom == 0] = 1.0
        corr = (x_reg.T @ x_target) / denom
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        w = torch.tensor(corr.T * scale, dtype=torch.float32, device=self.module.W_global.device)
        self.module.W_global.copy_(w * self.module.M_GRN)

    @torch.no_grad()
    def get_latent_time(self, adata: AnnData = None, n_samples: int = 1) -> np.ndarray:
        logger.warning("SPARC does not infer pseudo-time. Returning zeros.")
        return np.zeros(self.adata.n_obs)

    @torch.no_grad()
    def get_expression_velocity(self, adata: AnnData = None) -> np.ndarray:
        logger.warning("SPARC is steady-state. Returning alpha-like fitted rates.")
        self._check_if_trained()
        adata = self._validate_anndata(adata)
        device = next(self.module.parameters()).device
        scdl = self._make_data_loader(adata=adata, batch_size=256)
        alphas = []
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY].to(device)
            x_niche = tensors.get(REGISTRY_KEYS.X_NICHE_KEY)
            if x_niche is not None:
                x_niche = x_niche.to(device)
            inf = self.module.inference(x=x, x_niche=x_niche)
            gen = self.module.generative(
                x=x,
                z_intra=inf["z_intra"],
                x_niche=x_niche,
            )
            alpha = gen["x_hat"]
            alphas.append(alpha.cpu().numpy())
        return np.concatenate(alphas, axis=0)

    get_velocity = get_expression_velocity

    @torch.no_grad()
    def get_grn(self, adata: AnnData = None, method: str = "weight") -> pd.DataFrame:
        self._check_if_trained()
        if method != "weight":
            raise ValueError("SPARC currently supports method='weight' only.")
        w_effective = self.module._effective_global_grn_weights()
        w = w_effective.detach().cpu().numpy().T
        target_genes = np.array(self.adata.var_names)[self.module.target_index]
        reg_genes = np.array(self.adata.var_names)[self.module.regulator_index]
        return pd.DataFrame(w, index=reg_genes, columns=target_genes)

    def get_global_grn(self, adata: AnnData = None) -> pd.DataFrame:
        """Return the learned global TF-by-target regulatory weight matrix."""
        return self.get_grn(adata=adata, method="weight")

    @torch.no_grad()
    def get_cell_specific_grn(self, adata: AnnData = None) -> np.ndarray:
        self._check_if_trained()
        adata = self._validate_anndata(adata)
        device = next(self.module.parameters()).device
        scdl = self._make_data_loader(adata=adata, batch_size=256)
        all_w = []
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY].to(device)
            x_niche = tensors.get(REGISTRY_KEYS.X_NICHE_KEY)
            if x_niche is not None:
                x_niche = x_niche.to(device)
            inf = self.module.inference(x=x, x_niche=x_niche)
            gen = self.module.generative(
                x=x,
                z_intra=inf["z_intra"],
                x_niche=x_niche,
            )
            w_dynamic = gen["W_dynamic"]
            w_dynamic_t = w_dynamic.transpose(1, 2)
            all_w.append(w_dynamic_t.detach().cpu().numpy())
        return np.concatenate(all_w, axis=0)

    def get_dynamic_grn(self, adata: AnnData = None) -> np.ndarray:
        """Return cell-specific GRN weights with shape cell by TF by target."""
        return self.get_cell_specific_grn(adata=adata)

    @torch.no_grad()
    def get_spatial_communication(self) -> pd.DataFrame:
        if self.module.M_LR is None:
            raise ValueError("Model was not trained with a ligand-receptor mask.")
        effective_lambda = self.module._effective_lr_weights().detach().cpu().numpy()
        return pd.DataFrame(
            effective_lambda,
            index=self.adata.var_names,
            columns=self.adata.var_names,
        )

    @torch.no_grad()
    def get_spatial_effects(self) -> dict:
        """Return learned spatial cascade parameters as gene-labeled tables."""
        self._check_if_trained()
        gene_names = np.array(self.adata.var_names)
        reg_genes = gene_names[self.module.regulator_index]
        target_genes = gene_names[self.module.target_index]

        lambda_lr = self.module._effective_lr_weights().detach().cpu().numpy()
        omega = self.module._effective_omega_weights().detach().cpu().numpy()

        return {
            "lambda_lr": pd.DataFrame(lambda_lr, index=gene_names, columns=gene_names),
            "omega_receptor_to_tf": pd.DataFrame(omega, index=gene_names, columns=reg_genes),
        }

    @torch.no_grad()
    def get_top_grn_edges(
        self,
        n: int = 100,
        min_abs_weight: float = 0.0,
        include_self_loops: bool = False,
    ) -> pd.DataFrame:
        """Return top absolute global TF-target edges."""
        grn = self.get_grn()
        rows = []
        for tf, row in grn.iterrows():
            vals = row.to_numpy(dtype=np.float64)
            targets = grn.columns.to_numpy()
            for target, weight in zip(targets, vals):
                if not include_self_loops and tf == target:
                    continue
                abs_weight = abs(float(weight))
                if abs_weight > min_abs_weight:
                    rows.append(
                        {
                            "tf": tf,
                            "target": target,
                            "weight": float(weight),
                            "abs_weight": abs_weight,
                        }
                    )
        if not rows:
            return pd.DataFrame(columns=["tf", "target", "weight", "abs_weight"])
        return pd.DataFrame(rows).sort_values("abs_weight", ascending=False).head(n)

    @torch.no_grad()
    def get_top_cascade_paths(
        self,
        n: int = 200,
        min_abs_score: float = 0.0,
        include_tf_self_loops: bool = False,
        adata: AnnData = None,
        score_mode: str = "parameter",
        collapse_targets: bool = False,
        max_paths_per_tf: Optional[int] = None,
        max_paths_per_receptor: Optional[int] = None,
        max_paths_per_lr_pair: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return top L-R-TF-target path scores supported by the learned cascade.

        The parameter score is ``lambda_lr * omega_receptor_to_tf * W_global``.
        When ``score_mode='activity'``, the ranking score additionally uses
        the observed spatial LR activity. When ``score_mode='balanced_activity'``,
        LR, RTF, and GRN edge strengths are symmetrically degree-normalized
        before computing the path ranking score. This is a display/ranking
        normalization for functional path summaries, not a training constraint.
        These are compact functional-consistency summaries, not proofs that a
        biochemical path is active in every cell.
        """
        self._check_if_trained()
        if score_mode not in {"parameter", "activity", "balanced_activity"}:
            raise ValueError(
                "score_mode must be 'parameter', 'activity', or 'balanced_activity'."
            )
        if self.module.M_LR is None or self.module.M_RTF is None:
            return pd.DataFrame(
                columns=[
                    "ligand",
                    "receptor",
                    "tf",
                    "target",
                    "lambda_lr",
                    "effective_lr",
                    "omega",
                    "grn_weight",
                    "score_mode",
                    "lr_activity",
                    "path_score",
                    "abs_path_score",
                    "activity_weighted_score",
                    "balanced_lr_score",
                    "balanced_omega_score",
                    "balanced_grn_score",
                    "ranking_score",
                ]
            )

        uses_activity = score_mode in {"activity", "balanced_activity"}
        if uses_activity:
            adata = self._validate_anndata(adata)
            x_raw = self._get_registered_x(adata)
            x = x_raw.toarray() if sp.issparse(x_raw) else np.asarray(x_raw)
            x = x.astype(np.float32, copy=False)
            if "X_niche" in adata.obsm:
                x_niche = np.asarray(adata.obsm["X_niche"], dtype=np.float32)
            else:
                x_niche = x
            if self.module.lr_niche_contrast and "X_niche" in adata.obsm:
                ligand_activity_input = x_niche - x
            else:
                ligand_activity_input = x_niche
        else:
            x = None
            x_niche = None
            ligand_activity_input = None

        gene_names = np.array(self.adata.var_names)
        reg_genes = gene_names[self.module.regulator_index]
        target_genes = gene_names[self.module.target_index]

        lambda_lr = self.module._effective_lr_weights().detach().cpu().numpy()
        omega = self.module._effective_omega_weights().detach().cpu().numpy()
        grn = self.module._effective_global_grn_weights().detach().cpu().numpy().T

        balanced_lr = balanced_omega = balanced_grn = None
        lr_activity_matrix = None
        if score_mode == "balanced_activity":
            lr_activity_matrix = np.zeros_like(lambda_lr, dtype=np.float32)
            lr_pairs_for_activity = np.argwhere(lambda_lr != 0)
            for ligand_idx, receptor_idx in lr_pairs_for_activity:
                lr_activity_matrix[ligand_idx, receptor_idx] = float(
                    np.mean(ligand_activity_input[:, ligand_idx] * x[:, receptor_idx])
                )

            effective_lambda = lambda_lr.copy()
            if self.module.lr_degree_normalize and self.module.M_LR is not None:
                lr_degree = self.module.lr_receptor_degree.detach().cpu().numpy()
                effective_lambda = effective_lambda / np.maximum(lr_degree[None, :], 1.0)

            def _sym_norm(values: np.ndarray) -> np.ndarray:
                strength = np.abs(values).astype(np.float64, copy=False)
                row_sum = strength.sum(axis=1)
                col_sum = strength.sum(axis=0)
                denom = np.sqrt(
                    np.maximum(row_sum[:, None], 1e-12)
                    * np.maximum(col_sum[None, :], 1e-12)
                )
                return np.divide(
                    strength,
                    denom,
                    out=np.zeros_like(strength, dtype=np.float64),
                    where=denom > 0,
                ).astype(np.float32)

            balanced_lr = _sym_norm(effective_lambda * np.abs(lr_activity_matrix))
            balanced_omega = _sym_norm(omega)
            balanced_grn = _sym_norm(grn)

        rows = []
        lr_pairs = np.argwhere(lambda_lr != 0)
        for ligand_idx, receptor_idx in lr_pairs:
            lr_weight = float(lambda_lr[ligand_idx, receptor_idx])
            if self.module.lr_degree_normalize and self.module.M_LR is not None:
                lr_degree = float(self.module.lr_receptor_degree[receptor_idx].detach().cpu())
            else:
                lr_degree = 1.0
            effective_lr = lr_weight / max(lr_degree, 1.0)
            if uses_activity:
                if lr_activity_matrix is not None:
                    lr_activity = float(lr_activity_matrix[ligand_idx, receptor_idx])
                else:
                    lr_activity = float(
                        np.mean(ligand_activity_input[:, ligand_idx] * x[:, receptor_idx])
                    )
            else:
                lr_activity = 1.0
            tf_idx = np.flatnonzero(omega[receptor_idx] != 0)
            if tf_idx.size == 0:
                continue
            for local_tf_idx in tf_idx:
                omega_weight = float(omega[receptor_idx, local_tf_idx])
                target_idx = np.flatnonzero(grn[local_tf_idx] != 0)
                if target_idx.size == 0:
                    continue
                scores = effective_lr * omega_weight * grn[local_tf_idx, target_idx]
                for local_target_idx, score in zip(target_idx, scores):
                    if (
                        not include_tf_self_loops
                        and reg_genes[local_tf_idx] == target_genes[local_target_idx]
                    ):
                        continue
                    abs_score = abs(float(score))
                    if abs_score <= min_abs_score:
                        continue
                    activity_weighted_score = abs_score * abs(lr_activity)
                    balanced_lr_score = np.nan
                    balanced_omega_score = np.nan
                    balanced_grn_score = np.nan
                    if score_mode == "balanced_activity":
                        balanced_lr_score = float(balanced_lr[ligand_idx, receptor_idx])
                        balanced_omega_score = float(
                            balanced_omega[receptor_idx, local_tf_idx]
                        )
                        balanced_grn_score = float(
                            balanced_grn[local_tf_idx, local_target_idx]
                        )
                        ranking_score = (
                            balanced_lr_score
                            * balanced_omega_score
                            * balanced_grn_score
                        )
                    elif score_mode == "activity":
                        ranking_score = activity_weighted_score
                    else:
                        ranking_score = abs_score
                    rows.append(
                        {
                            "ligand": gene_names[ligand_idx],
                            "receptor": gene_names[receptor_idx],
                            "tf": reg_genes[local_tf_idx],
                            "target": target_genes[local_target_idx],
                            "lambda_lr": lr_weight,
                            "effective_lr": effective_lr,
                            "omega": omega_weight,
                            "grn_weight": float(grn[local_tf_idx, local_target_idx]),
                            "score_mode": score_mode,
                            "lr_activity": lr_activity,
                            "path_score": float(score),
                            "abs_path_score": abs_score,
                            "activity_weighted_score": activity_weighted_score,
                            "balanced_lr_score": balanced_lr_score,
                            "balanced_omega_score": balanced_omega_score,
                            "balanced_grn_score": balanced_grn_score,
                            "ranking_score": ranking_score,
                        }
                    )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "ligand",
                    "receptor",
                    "tf",
                    "target",
                    "lambda_lr",
                    "effective_lr",
                    "omega",
                    "grn_weight",
                    "score_mode",
                    "lr_activity",
                    "path_score",
                    "abs_path_score",
                    "activity_weighted_score",
                    "balanced_lr_score",
                    "balanced_omega_score",
                    "balanced_grn_score",
                    "ranking_score",
                ]
            )
        df = pd.DataFrame(rows).sort_values("ranking_score", ascending=False)
        if collapse_targets:
            df = (
                df.sort_values("ranking_score", ascending=False)
                .groupby(["ligand", "receptor", "tf"], as_index=False, sort=False)
                .first()
                .sort_values("ranking_score", ascending=False)
            )
        if max_paths_per_tf is not None or max_paths_per_receptor is not None or max_paths_per_lr_pair is not None:
            selected = []
            tf_counts: dict[str, int] = {}
            receptor_counts: dict[str, int] = {}
            lr_counts: dict[tuple[str, str], int] = {}
            for _, row in df.iterrows():
                tf = row["tf"]
                receptor = row["receptor"]
                lr_pair = (row["ligand"], row["receptor"])
                if max_paths_per_tf is not None and tf_counts.get(tf, 0) >= max_paths_per_tf:
                    continue
                if (
                    max_paths_per_receptor is not None
                    and receptor_counts.get(receptor, 0) >= max_paths_per_receptor
                ):
                    continue
                if (
                    max_paths_per_lr_pair is not None
                    and lr_counts.get(lr_pair, 0) >= max_paths_per_lr_pair
                ):
                    continue
                selected.append(row)
                tf_counts[tf] = tf_counts.get(tf, 0) + 1
                receptor_counts[receptor] = receptor_counts.get(receptor, 0) + 1
                lr_counts[lr_pair] = lr_counts.get(lr_pair, 0) + 1
                if len(selected) >= n:
                    break
            return pd.DataFrame(selected)
        return df.head(n)

    @torch.no_grad()
    def get_spatial_diagnostics(self, adata: AnnData = None) -> pd.DataFrame:
        """Summarize whether the learned spatial cascade is active."""
        self._check_if_trained()
        adata = self._validate_anndata(adata)
        device = next(self.module.parameters()).device
        scdl = self._make_data_loader(adata=adata, batch_size=256)

        rows = []
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY].to(device)
            x_niche = tensors.get(REGISTRY_KEYS.X_NICHE_KEY)
            if x_niche is not None:
                x_niche = x_niche.to(device)
            inf = self.module.inference(x=x, x_niche=x_niche)
            gen = self.module.generative(
                x=x,
                z_intra=inf["z_intra"],
                x_niche=x_niche,
            )
            rows.append(
                {
                    "mean_abs_receptor_signal": gen["receptor_signal"].abs().mean().item(),
                    "mean_abs_delta_tf_activity": gen["delta_a"].abs().mean().item(),
                    "mean_abs_delta_tf_activity_lr": gen["delta_a_lr_raw"].abs().mean().item(),
                    "mean_abs_delta_tf_activity_pathway": gen["delta_a_pathway"].abs().mean().item(),
                    "mean_abs_delta_w_state": gen["delta_W_state"].abs().mean().item(),
                }
            )

        diagnostics = pd.DataFrame(rows).mean(axis=0).to_dict()

        w_eff = self.module._effective_global_grn_weights().detach()
        grn_tf_load = torch.sum(torch.abs(w_eff), dim=0)
        active_tf = grn_tf_load > self.module.omega_norm_eps
        if torch.any(active_tf):
            diagnostics["grn_tf_load_max_over_mean"] = (
                grn_tf_load[active_tf].max()
                / grn_tf_load[active_tf].mean().clamp_min(self.module.omega_norm_eps)
            ).item()
        else:
            diagnostics["grn_tf_load_max_over_mean"] = 0.0

        omega = self.module._effective_omega_weights().detach().abs()
        omega_row_sum = omega.sum(dim=1)
        active_rows = omega_row_sum > self.module.omega_norm_eps
        if torch.any(active_rows):
            row_max_share = omega[active_rows].max(dim=1).values / omega_row_sum[
                active_rows
            ].clamp_min(self.module.omega_norm_eps)
            diagnostics["omega_row_max_share_mean"] = row_max_share.mean().item()
            diagnostics["omega_row_max_share_max"] = row_max_share.max().item()
        else:
            diagnostics["omega_row_max_share_mean"] = 0.0
            diagnostics["omega_row_max_share_max"] = 0.0

        lambda_lr = self.module._effective_lr_weights().detach().abs()
        lr_receptor_load = lambda_lr.sum(dim=0)
        active_receptors = lr_receptor_load > self.module.omega_norm_eps
        if torch.any(active_receptors):
            diagnostics["lr_receptor_load_max_over_mean"] = (
                lr_receptor_load[active_receptors].max()
                / lr_receptor_load[active_receptors].mean().clamp_min(self.module.omega_norm_eps)
            ).item()
        else:
            diagnostics["lr_receptor_load_max_over_mean"] = 0.0

        return pd.Series(diagnostics).to_frame("value")

    @torch.no_grad()
    def predict_perturbation(
        self,
        adata: AnnData = None,
        gene: str = None,
        ko_value: float = 0.0,
        cell_mask: Optional[np.ndarray] = None,
        n_steps: int = 10,
        tol: float = 1e-4,
        damping: float = 0.5,
        return_trajectory: bool = False,
        batch_size: int = 2048,
    ):
        """In-silico perturbation via fixed-point rollout on the trained graph.

        Parameters
        ----------
        cell_mask
            Optional boolean mask selecting cells where the intervention is
            applied. When omitted, the perturbation is global, matching the
            historical behavior.
        """
        self._check_if_trained()
        adata = self._validate_anndata(adata)
        device = next(self.module.parameters()).device

        # Ensure deterministic forward passes: disable dropout and use
        # running BatchNorm statistics so identical inputs always produce
        # identical outputs (prevents stochastic noise in remote cells).
        was_training = self.module.training
        self.module.eval()

        if gene is None:
            raise ValueError("gene must be provided for perturbation prediction.")
        if gene not in adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.var_names.")

        global_idx = list(adata.var_names).index(gene)
        if cell_mask is None:
            cell_mask_np = np.ones(adata.n_obs, dtype=bool)
        else:
            cell_mask_np = np.asarray(cell_mask, dtype=bool)
            if cell_mask_np.shape[0] != adata.n_obs:
                raise ValueError(
                    f"cell_mask length {cell_mask_np.shape[0]} does not match n_obs {adata.n_obs}."
                )
            if cell_mask_np.sum() == 0:
                raise ValueError("cell_mask selects no cells for perturbation.")
        cell_mask_t = torch.tensor(cell_mask_np, dtype=torch.bool, device=device)

        a_sparse = None
        if "spatial_connectivities" in adata.obsp:
            a = sp.coo_matrix(adata.obsp["spatial_connectivities"])
            indices = np.vstack((a.row, a.col))
            a_sparse = torch.sparse_coo_tensor(
                torch.LongTensor(indices),
                torch.FloatTensor(a.data),
                torch.Size(a.shape),
            ).coalesce().to(device)

        x_matrix = self._get_registered_x(adata)
        x_raw = x_matrix.toarray() if sp.issparse(x_matrix) else np.asarray(x_matrix)
        x_base = torch.tensor(x_raw, dtype=torch.float32, device=device)

        def compute_niche(x_input):
            if a_sparse is None:
                return x_input
            return torch.sparse.mm(a_sparse, x_input)

        def get_x_hat(x_input):
            x_niche = compute_niche(x_input)
            x_hat_list = []
            for i in range(0, x_input.shape[0], batch_size):
                x_batch = x_input[i : i + batch_size]
                x_niche_batch = x_niche[i : i + batch_size]
                inf = self.module.inference(x=x_batch, x_niche=x_niche_batch, n_samples=1)
                # Use encoder means (deterministic) instead of stochastic
                # samples to ensure identical inputs produce identical outputs.
                gen = self.module.generative(
                    x=x_batch,
                    z_intra=inf["qz_m_intra"],
                    x_niche=x_niche_batch,
                )
                x_hat_list.append(gen["x_hat"])
            return torch.cat(x_hat_list, dim=0)

        x_hat_base = get_x_hat(x_base)

        x_iter = x_base.clone()
        x_iter[cell_mask_t, global_idx] = ko_value
        trajectory = []

        target_mask = torch.tensor(self.module.target_index, device=device, dtype=torch.bool)
        x_base_targets = x_base[:, target_mask]
        delta = torch.zeros((adata.n_obs, int(target_mask.sum())), device=device)

        # Feedback mask: only cells within this mask get their expression
        # updated in x_iter. Starts with KO cells and expands by one spatial
        # hop per rollout step, correctly modeling cascade propagation.
        # Remote cells keep x_base expression, preventing VAE encoder input
        # drift that would otherwise cause delta to grow with distance.
        feedback_mask = cell_mask_t.clone()

        for _ in range(n_steps):
            x_hat_ko = get_x_hat(x_iter)
            new_delta = x_hat_ko - x_hat_base
            relaxed_delta = damping * new_delta + (1.0 - damping) * delta

            updated_targets = torch.clamp(
                x_base_targets + relaxed_delta,
                min=0.0,
            )
            # Only write delta back for cells within propagation range;
            # remote cells retain x_base expression to avoid encoder drift.
            x_iter[:, target_mask] = torch.where(
                feedback_mask.unsqueeze(-1),
                updated_targets,
                x_base_targets,
            )
            x_iter[cell_mask_t, global_idx] = ko_value

            if return_trajectory:
                trajectory.append(relaxed_delta.cpu().numpy())

            if torch.norm(relaxed_delta - delta) < tol:
                delta = relaxed_delta
                break
            delta = relaxed_delta

            # Expand feedback mask by one spatial hop for the next iteration,
            # allowing the perturbation cascade to propagate outward.
            if a_sparse is not None:
                neighbor_signal = torch.sparse.mm(
                    a_sparse, feedback_mask.float().unsqueeze(-1)
                ).squeeze(-1)
                feedback_mask = feedback_mask | (neighbor_signal > 0)

        # Restore original training state.
        if was_training:
            self.module.train()

        if return_trajectory:
            return trajectory
        return delta.cpu().numpy()

    @torch.no_grad()
    def explain_perturbation_response(
        self,
        adata: AnnData = None,
        gene: str = None,
        ko_value: float = 0.0,
        cell_mask: Optional[np.ndarray] = None,
        n_steps: int = 10,
        tol: float = 1e-4,
        damping: float = 0.5,
        top_response_genes: int = 5,
        response_cell_fraction: float = 0.1,
        min_response_cells: int = 20,
        max_response_cells: int = 128,
        max_paths_per_gene: int = 100,
        batch_size: int = 512,
    ) -> dict:
        """Trace local LR-RTF-GRN paths for top perturbation responses.

        This routine reruns the same deterministic perturbation rollout used by
        :meth:`predict_perturbation`, then explains the largest target-gene
        responses in the cells where each response is strongest.  For a path
        ``ligand -> receptor -> TF -> target``, the reported contribution is
        the mean local contribution to the decoder logit from the perturbation-
        induced change in that LR edge signal:

        ``Delta LR_edge_signal * Omega[receptor, TF] * midpoint(W_dynamic[target, TF])``.

        The decomposition uses the trained forward graph and fixed learned
        parameters.  It is intentionally local to a perturbation result; global
        cascade-path rankings remain only a coarse diagnostic.
        """
        self._check_if_trained()
        adata = self._validate_anndata(adata)
        device = next(self.module.parameters()).device

        if gene is None:
            raise ValueError("gene must be provided for perturbation explanation.")
        if gene not in adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.var_names.")
        if not 0.0 < response_cell_fraction <= 1.0:
            raise ValueError("response_cell_fraction must be in (0, 1].")
        if top_response_genes < 1:
            raise ValueError("top_response_genes must be >= 1.")
        if min_response_cells < 1:
            raise ValueError("min_response_cells must be >= 1.")
        if max_response_cells < min_response_cells:
            raise ValueError("max_response_cells must be >= min_response_cells.")

        global_idx = list(adata.var_names).index(gene)
        if cell_mask is None:
            cell_mask_np = np.ones(adata.n_obs, dtype=bool)
        else:
            cell_mask_np = np.asarray(cell_mask, dtype=bool)
            if cell_mask_np.shape[0] != adata.n_obs:
                raise ValueError(
                    f"cell_mask length {cell_mask_np.shape[0]} does not match n_obs {adata.n_obs}."
                )
            if cell_mask_np.sum() == 0:
                raise ValueError("cell_mask selects no cells for perturbation.")
        cell_mask_t = torch.tensor(cell_mask_np, dtype=torch.bool, device=device)

        a_sparse = None
        if "spatial_connectivities" in adata.obsp:
            a = sp.coo_matrix(adata.obsp["spatial_connectivities"])
            indices = np.vstack((a.row, a.col))
            a_sparse = torch.sparse_coo_tensor(
                torch.LongTensor(indices),
                torch.FloatTensor(a.data),
                torch.Size(a.shape),
            ).coalesce().to(device)

        x_matrix = self._get_registered_x(adata)
        x_raw = x_matrix.toarray() if sp.issparse(x_matrix) else np.asarray(x_matrix)
        x_base = torch.tensor(x_raw, dtype=torch.float32, device=device)

        def compute_niche(x_input: torch.Tensor) -> torch.Tensor:
            if a_sparse is None:
                return x_input
            return torch.sparse.mm(a_sparse, x_input)

        def get_x_hat(x_input: torch.Tensor) -> torch.Tensor:
            x_niche = compute_niche(x_input)
            x_hat_list = []
            for i in range(0, x_input.shape[0], batch_size):
                x_batch = x_input[i : i + batch_size]
                x_niche_batch = x_niche[i : i + batch_size]
                inf = self.module.inference(x=x_batch, x_niche=x_niche_batch, n_samples=1)
                gen = self.module.generative(
                    x=x_batch,
                    z_intra=inf["qz_m_intra"],
                    x_niche=x_niche_batch,
                )
                x_hat_list.append(gen["x_hat"])
            return torch.cat(x_hat_list, dim=0)

        def run_components(
            x_input: torch.Tensor,
            x_niche: torch.Tensor,
            indices_np: np.ndarray,
        ) -> dict[str, np.ndarray]:
            chunks = {
                "x_hat": [],
                "target_logits": [],
                "W_dynamic": [],
                "delta_a_raw": [],
                "a_tilde": [],
            }
            for i in range(0, indices_np.size, batch_size):
                idx = torch.tensor(indices_np[i : i + batch_size], dtype=torch.long, device=device)
                x_batch = x_input.index_select(0, idx)
                x_niche_batch = x_niche.index_select(0, idx)
                inf = self.module.inference(x=x_batch, x_niche=x_niche_batch, n_samples=1)
                gen = self.module.generative(
                    x=x_batch,
                    z_intra=inf["qz_m_intra"],
                    x_niche=x_niche_batch,
                )
                for key in chunks:
                    chunks[key].append(gen[key].detach().cpu().numpy())
            return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}

        was_training = self.module.training
        self.module.eval()

        x_hat_base = get_x_hat(x_base)
        x_iter = x_base.clone()
        x_iter[cell_mask_t, global_idx] = ko_value

        target_mask = torch.tensor(self.module.target_index, device=device, dtype=torch.bool)
        x_base_targets = x_base[:, target_mask]
        delta = torch.zeros((adata.n_obs, int(target_mask.sum())), device=device)
        feedback_mask = cell_mask_t.clone()

        for _ in range(n_steps):
            x_hat_ko = get_x_hat(x_iter)
            new_delta = x_hat_ko - x_hat_base
            relaxed_delta = damping * new_delta + (1.0 - damping) * delta

            updated_targets = torch.clamp(x_base_targets + relaxed_delta, min=0.0)
            x_iter[:, target_mask] = torch.where(
                feedback_mask.unsqueeze(-1),
                updated_targets,
                x_base_targets,
            )
            x_iter[cell_mask_t, global_idx] = ko_value

            if torch.norm(relaxed_delta - delta) < tol:
                delta = relaxed_delta
                break
            delta = relaxed_delta

            if a_sparse is not None:
                neighbor_signal = torch.sparse.mm(
                    a_sparse, feedback_mask.float().unsqueeze(-1)
                ).squeeze(-1)
                feedback_mask = feedback_mask | (neighbor_signal > 0)

        x_final = x_iter
        x_niche_base = compute_niche(x_base)
        x_niche_final = compute_niche(x_final)

        if was_training:
            self.module.train()

        delta_np = delta.detach().cpu().numpy()
        target_genes = np.array(adata.var_names)[self.module.target_index]
        response_summary = pd.DataFrame(
            {
                "response_rank": np.arange(1, len(target_genes) + 1),
                "gene": target_genes,
                "mean_delta": delta_np.mean(axis=0),
                "mean_abs_delta": np.abs(delta_np).mean(axis=0),
                "max_abs_delta": np.abs(delta_np).max(axis=0),
            }
        ).sort_values("mean_abs_delta", ascending=False, ignore_index=True)
        response_summary["response_rank"] = np.arange(1, response_summary.shape[0] + 1)

        if self.module.M_LR is None or self.module.M_RTF is None:
            empty_paths = pd.DataFrame()
            return {
                "paths": empty_paths,
                "response_genes": response_summary.head(top_response_genes),
                "tf_summary": pd.DataFrame(),
                "lr_summary": pd.DataFrame(),
                "cell_groups": pd.DataFrame(),
                "delta": delta_np,
            }

        gene_names = np.array(self.adata.var_names)
        reg_genes = gene_names[self.module.regulator_index]
        lambda_lr = self.module._effective_lr_weights().detach().cpu().numpy()
        omega = self.module._effective_omega_weights().detach().cpu().numpy()
        m_grn = self.module.M_GRN.detach().cpu().numpy()
        lr_pairs = np.argwhere(lambda_lr != 0)
        rtf_by_receptor = {
            int(r): np.flatnonzero(omega[int(r)] != 0) for r in np.unique(lr_pairs[:, 1])
        }
        if self.module.lr_degree_normalize and self.module.M_LR is not None:
            lr_degree = self.module.lr_receptor_degree.detach().cpu().numpy()
        else:
            lr_degree = np.ones(adata.n_vars, dtype=np.float32)

        all_path_rows = []
        gene_rows = []
        cell_group_rows = []
        obs_names = np.asarray(adata.obs_names)

        for _, gene_row in response_summary.head(top_response_genes).iterrows():
            response_gene = str(gene_row["gene"])
            target_idx = int(np.where(target_genes == response_gene)[0][0])
            target_delta = delta_np[:, target_idx]
            order = np.argsort(-np.abs(target_delta))
            n_cells = int(
                min(
                    max_response_cells,
                    max(min_response_cells, np.ceil(response_cell_fraction * adata.n_obs)),
                )
            )
            n_cells = min(n_cells, adata.n_obs)
            selected_idx = np.sort(order[:n_cells].astype(int))

            comp_base = run_components(x_base, x_niche_base, selected_idx)
            comp_final = run_components(x_final, x_niche_final, selected_idx)

            x_base_sel = x_base[selected_idx].detach().cpu().numpy()
            x_final_sel = x_final[selected_idx].detach().cpu().numpy()
            niche_base_sel = x_niche_base[selected_idx].detach().cpu().numpy()
            niche_final_sel = x_niche_final[selected_idx].detach().cpu().numpy()
            if self.module.lr_niche_contrast:
                ligand_base = niche_base_sel - x_base_sel
                ligand_final = niche_final_sel - x_final_sel
            else:
                ligand_base = niche_base_sel
                ligand_final = niche_final_sel

            w_mid_target = 0.5 * (
                comp_base["W_dynamic"][:, target_idx, :]
                + comp_final["W_dynamic"][:, target_idx, :]
            )
            logits_delta = (
                comp_final["target_logits"][:, target_idx]
                - comp_base["target_logits"][:, target_idx]
            )
            xhat_delta = comp_final["x_hat"][:, target_idx] - comp_base["x_hat"][:, target_idx]
            selected_delta = target_delta[selected_idx]

            cell_group_rows.append(
                {
                    "response_gene": response_gene,
                    "response_rank": int(gene_row["response_rank"]),
                    "n_selected_spots": int(selected_idx.size),
                    "selection": "largest absolute predicted response per gene",
                    "response_cell_fraction": float(response_cell_fraction),
                    "min_response_cells": int(min_response_cells),
                    "max_response_cells": int(max_response_cells),
                    "mean_delta_selected": float(np.mean(selected_delta)),
                    "mean_abs_delta_selected": float(np.mean(np.abs(selected_delta))),
                    "mean_target_logit_delta_selected": float(np.mean(logits_delta)),
                    "mean_xhat_delta_selected": float(np.mean(xhat_delta)),
                    "selected_obs_names": ";".join(obs_names[selected_idx].astype(str)),
                }
            )

            target_supported_tfs = np.flatnonzero(m_grn[target_idx] != 0)
            target_supported_tf_set = set(int(x) for x in target_supported_tfs)
            gene_path_rows = []
            for ligand_idx, receptor_idx in lr_pairs:
                ligand_idx = int(ligand_idx)
                receptor_idx = int(receptor_idx)
                tf_idx = [
                    int(tf)
                    for tf in rtf_by_receptor.get(receptor_idx, np.array([], dtype=int))
                    if int(tf) in target_supported_tf_set
                ]
                if not tf_idx:
                    continue
                effective_lr = float(lambda_lr[ligand_idx, receptor_idx]) / max(
                    float(lr_degree[receptor_idx]), 1.0
                )
                base_edge_signal = (
                    ligand_base[:, ligand_idx] * effective_lr * x_base_sel[:, receptor_idx]
                )
                final_edge_signal = (
                    ligand_final[:, ligand_idx] * effective_lr * x_final_sel[:, receptor_idx]
                )
                delta_edge_signal = final_edge_signal - base_edge_signal
                if not np.any(delta_edge_signal):
                    continue
                for local_tf_idx in tf_idx:
                    omega_weight = float(omega[receptor_idx, local_tf_idx])
                    if omega_weight == 0.0:
                        continue
                    cell_contrib = (
                        delta_edge_signal
                        * omega_weight
                        * w_mid_target[:, local_tf_idx]
                    )
                    mean_contrib = float(np.mean(cell_contrib))
                    mean_abs_contrib = float(np.mean(np.abs(cell_contrib)))
                    if mean_abs_contrib <= 0.0:
                        continue
                    gene_path_rows.append(
                        {
                            "response_gene": response_gene,
                            "response_rank": int(gene_row["response_rank"]),
                            "perturbation_gene": gene,
                            "n_selected_spots": int(selected_idx.size),
                            "ligand": str(gene_names[ligand_idx]),
                            "receptor": str(gene_names[receptor_idx]),
                            "tf": str(reg_genes[local_tf_idx]),
                            "target": response_gene,
                            "lambda_lr": float(lambda_lr[ligand_idx, receptor_idx]),
                            "effective_lr": effective_lr,
                            "omega": omega_weight,
                            "mean_midpoint_dynamic_grn": float(
                                np.mean(w_mid_target[:, local_tf_idx])
                            ),
                            "mean_base_lr_edge_signal": float(np.mean(base_edge_signal)),
                            "mean_final_lr_edge_signal": float(np.mean(final_edge_signal)),
                            "mean_delta_lr_edge_signal": float(np.mean(delta_edge_signal)),
                            "mean_decoder_logit_path_contribution": mean_contrib,
                            "mean_abs_decoder_logit_path_contribution": mean_abs_contrib,
                            "path_direction": "positive" if mean_contrib >= 0 else "negative",
                        }
                    )

            if gene_path_rows:
                gene_paths = pd.DataFrame(gene_path_rows)
                full_sum = float(gene_paths["mean_decoder_logit_path_contribution"].sum())
                gene_paths = gene_paths.sort_values(
                    "mean_abs_decoder_logit_path_contribution",
                    ascending=False,
                    ignore_index=True,
                )
                gene_paths.insert(2, "path_rank_within_response_gene", np.arange(1, gene_paths.shape[0] + 1))
                if max_paths_per_gene > 0:
                    gene_paths = gene_paths.head(max_paths_per_gene).copy()
                all_path_rows.append(gene_paths)
            else:
                full_sum = 0.0

            gene_rows.append(
                {
                    "response_gene": response_gene,
                    "response_rank": int(gene_row["response_rank"]),
                    "mean_delta_all_spots": float(gene_row["mean_delta"]),
                    "mean_abs_delta_all_spots": float(gene_row["mean_abs_delta"]),
                    "n_selected_spots": int(selected_idx.size),
                    "mean_delta_selected": float(np.mean(selected_delta)),
                    "mean_abs_delta_selected": float(np.mean(np.abs(selected_delta))),
                    "mean_target_logit_delta_selected": float(np.mean(logits_delta)),
                    "mean_xhat_delta_selected": float(np.mean(xhat_delta)),
                    "sum_all_lr_rtf_grn_path_logit_contributions": full_sum,
                    "residual_logit_delta_after_lr_rtf_grn_paths": float(
                        np.mean(logits_delta) - full_sum
                    ),
                    "abs_lr_rtf_grn_path_fraction_of_logit_delta": float(
                        abs(full_sum) / (abs(float(np.mean(logits_delta))) + 1e-12)
                    ),
                    "attribution_note": (
                        "Path contributions decompose the perturbation-induced LR edge "
                        "signal through Omega and midpoint dynamic GRN weights; residual "
                        "target-logit change can include direct TF-expression shifts, "
                        "dynamic-GRN shifts, clamp effects, and softplus nonlinearity."
                    ),
                }
            )

        paths = (
            pd.concat(all_path_rows, ignore_index=True)
            if all_path_rows
            else pd.DataFrame(
                columns=[
                    "response_gene",
                    "response_rank",
                    "path_rank_within_response_gene",
                    "perturbation_gene",
                    "ligand",
                    "receptor",
                    "tf",
                    "target",
                    "mean_decoder_logit_path_contribution",
                    "mean_abs_decoder_logit_path_contribution",
                ]
            )
        )
        response_genes = pd.DataFrame(gene_rows)
        cell_groups = pd.DataFrame(cell_group_rows)

        if not paths.empty:
            tf_summary = (
                paths.groupby(["response_gene", "tf"], as_index=False)
                .agg(
                    mean_decoder_logit_path_contribution=(
                        "mean_decoder_logit_path_contribution",
                        "sum",
                    ),
                    mean_abs_decoder_logit_path_contribution=(
                        "mean_abs_decoder_logit_path_contribution",
                        "sum",
                    ),
                    n_paths=("tf", "size"),
                )
                .sort_values(
                    ["response_gene", "mean_abs_decoder_logit_path_contribution"],
                    ascending=[True, False],
                )
            )
            lr_summary = (
                paths.groupby(["response_gene", "ligand", "receptor"], as_index=False)
                .agg(
                    mean_decoder_logit_path_contribution=(
                        "mean_decoder_logit_path_contribution",
                        "sum",
                    ),
                    mean_abs_decoder_logit_path_contribution=(
                        "mean_abs_decoder_logit_path_contribution",
                        "sum",
                    ),
                    n_paths=("tf", "size"),
                )
                .sort_values(
                    ["response_gene", "mean_abs_decoder_logit_path_contribution"],
                    ascending=[True, False],
                )
            )
        else:
            tf_summary = pd.DataFrame()
            lr_summary = pd.DataFrame()

        return {
            "paths": paths,
            "response_genes": response_genes,
            "tf_summary": tf_summary,
            "lr_summary": lr_summary,
            "cell_groups": cell_groups,
            "delta": delta_np,
        }

    @torch.no_grad()
    def predict_microenvironment_transplant(
        self,
        adata: AnnData = None,
        donor_mask: Optional[np.ndarray] = None,
        environment_mask: Optional[np.ndarray] = None,
        x_niche: Optional[np.ndarray] = None,
        batch_size: int = 2048,
        return_dataframe: bool = False,
    ) -> dict:
        """Predict receiver-state changes after virtual microenvironment transfer.

        The donor cell's local expression and intrinsic latent state are kept
        fixed. Only the spatial niche input is replaced, either by a supplied
        ``x_niche`` matrix or by the mean log-expression profile of cells
        selected by ``environment_mask``. In v4, the replacement niche can
        influence predictions only through the LR -> RTF -> TF activity cascade.
        """
        self._check_if_trained()
        adata = self._validate_anndata(adata)
        device = next(self.module.parameters()).device

        if donor_mask is None:
            donor_mask_np = np.ones(adata.n_obs, dtype=bool)
        else:
            donor_mask_np = np.asarray(donor_mask)
            if donor_mask_np.dtype == bool:
                if donor_mask_np.shape[0] != adata.n_obs:
                    raise ValueError(
                        f"donor_mask length {donor_mask_np.shape[0]} does not match n_obs {adata.n_obs}."
                    )
            else:
                idx = donor_mask_np.astype(int)
                donor_mask_np = np.zeros(adata.n_obs, dtype=bool)
                donor_mask_np[idx] = True
        if donor_mask_np.sum() == 0:
            raise ValueError("donor_mask selects no cells.")

        x_matrix = self._get_registered_x(adata)
        x_raw = x_matrix.toarray() if sp.issparse(x_matrix) else np.asarray(x_matrix)
        y_expr = x_raw.astype(np.float32, copy=False)

        if "X_niche" in adata.obsm:
            baseline_niche = np.asarray(adata.obsm["X_niche"], dtype=np.float32)
        else:
            baseline_niche = y_expr

        donor_idx = np.flatnonzero(donor_mask_np)
        x_donor = x_raw[donor_idx].astype(np.float32, copy=False)
        baseline_niche_donor = baseline_niche[donor_idx].astype(np.float32, copy=False)

        if x_niche is not None:
            cf_niche = np.asarray(x_niche, dtype=np.float32)
            if cf_niche.ndim == 1:
                cf_niche = np.repeat(cf_niche[None, :], donor_idx.size, axis=0)
            if cf_niche.shape != (donor_idx.size, adata.n_vars):
                raise ValueError(
                    f"x_niche must have shape {(adata.n_vars,)} or {(donor_idx.size, adata.n_vars)}, "
                    f"got {cf_niche.shape}."
                )
        else:
            if environment_mask is None:
                raise ValueError("Provide either environment_mask or x_niche.")
            environment_mask_np = np.asarray(environment_mask)
            if environment_mask_np.dtype == bool:
                if environment_mask_np.shape[0] != adata.n_obs:
                    raise ValueError(
                        "environment_mask length "
                        f"{environment_mask_np.shape[0]} does not match n_obs {adata.n_obs}."
                    )
            else:
                env_idx = environment_mask_np.astype(int)
                environment_mask_np = np.zeros(adata.n_obs, dtype=bool)
                environment_mask_np[env_idx] = True
            if environment_mask_np.sum() == 0:
                raise ValueError("environment_mask selects no cells.")
            env_profile = y_expr[environment_mask_np].mean(axis=0).astype(np.float32)
            cf_niche = np.repeat(env_profile[None, :], donor_idx.size, axis=0)

        baseline_chunks = []
        transplanted_chunks = []
        was_training = self.module.training
        self.module.eval()
        for start in range(0, donor_idx.size, batch_size):
            stop = start + batch_size
            x_batch = torch.tensor(x_donor[start:stop], dtype=torch.float32, device=device)
            base_niche_batch = torch.tensor(
                baseline_niche_donor[start:stop], dtype=torch.float32, device=device
            )
            cf_niche_batch = torch.tensor(cf_niche[start:stop], dtype=torch.float32, device=device)

            inf_base = self.module.inference(x=x_batch, x_niche=base_niche_batch)
            gen_base = self.module.generative(
                x=x_batch,
                z_intra=inf_base["qz_m_intra"],
                x_niche=base_niche_batch,
            )
            gen_cf = self.module.generative(
                x=x_batch,
                z_intra=inf_base["qz_m_intra"],
                x_niche=cf_niche_batch,
            )
            baseline_chunks.append(gen_base["x_hat"].cpu().numpy())
            transplanted_chunks.append(gen_cf["x_hat"].cpu().numpy())

        if was_training:
            self.module.train()

        baseline = np.concatenate(baseline_chunks, axis=0)
        transplanted = np.concatenate(transplanted_chunks, axis=0)
        delta = transplanted - baseline
        target_genes = np.array(adata.var_names)[self.module.target_index]

        result = {
            "baseline": baseline,
            "transplanted": transplanted,
            "delta": delta,
            "donor_index": donor_idx,
            "target_genes": target_genes,
        }
        if return_dataframe:
            obs_names = adata.obs_names[donor_idx]
            result["baseline_df"] = pd.DataFrame(baseline, index=obs_names, columns=target_genes)
            result["transplanted_df"] = pd.DataFrame(
                transplanted, index=obs_names, columns=target_genes
            )
            result["delta_df"] = pd.DataFrame(delta, index=obs_names, columns=target_genes)
        return result

    predict_virtual_transplant = predict_microenvironment_transplant
