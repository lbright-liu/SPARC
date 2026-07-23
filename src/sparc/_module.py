"""SPARC VAE module with a single-latent spatial cascade decoder.

Current forward graph:
  local expression -> z -> dynamic GRN
  spatial niche -> ligand-receptor signal -> receptor-to-TF modulation
  dynamic GRN + modulated TF activity -> target reconstruction.

"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder

from ._constants import REGISTRY_KEYS


def _softplus_inverse(value: float) -> float:
    value_t = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(value_t).clamp_min(1e-8)))


class SPARCVAE(BaseModuleClass):
    """Single-latent pathway-constrained spatial GRN model.

    ``z_intra`` is inferred from the receiver cell's own expression and decodes
    the cell-state-specific GRN offset. Spatial niche expression is never
    encoded into a free latent variable; it only enters the decoder through
    ligand-receptor and receptor-to-TF parameters constrained by biological
    priors.
    """

    def __init__(
        self,
        n_input: int,
        regulator_index: list,
        target_index: list,
        skeleton: torch.Tensor,
        grn_prior_mask: torch.Tensor = None,
        lr_mask: np.ndarray = None,
        rec_tf_mask: np.ndarray = None,
        pathway_tf_mask: np.ndarray = None,
        encoder_input_mask: np.ndarray = None,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_layers: int = 1,
        lambda_struct: float = None,
        beta: float = 1.0,
        lam: float = 1.0,
        lam2: float = 1.0,
        lam_dyn: float = 1e-4,
        lam_omega: float = 1e-4,
        normalized_regularization: bool = True,
        dropout_rate: float = 0.1,
        lr_positive: bool = True,
        lr_init: float = 1e-2,
        lr_init_noise: float = 1e-3,
        communication_init: float = 1e-3,
        w_init: float = 1e-3,
        dynamic_scale_init: float = 0.15,
        omega_row_normalize: bool = True,
        omega_norm_eps: float = 1e-6,
        tf_activity_dropout: float = 0.0,
        lambda_omega_balance: float = 0.0,
        omega_balance_tau: float = 5.0,
        omega_balance_mode: str = "penalty",
        omega_balance_iters: int = 3,
        omega_row_max_share: float = 1.0,
        grn_tf_load_alpha: float = 0.0,
        grn_prior_penalty_weight: float = 0.1,
        grn_candidate_penalty_weight: float = 1.0,
        grn_disallowed_penalty_weight: float = 1.0,
        pathway_tf_penalty_weight: float = 0.5,
        lr_degree_normalize: bool = False,
        lr_niche_contrast: bool = False,
        encoder_niche_mode: str = "none",
        encoder_niche_mix_weight: float = 1.0,
        niche_effect_mode: str = "cascade",
        niche_residual_mode: str = "none",
        likelihood: str = "mse",
        disable_dynamic_grn: bool = False,
        fixed_lr_weights: np.ndarray = None,
        fixed_omega_weights: np.ndarray = None,
        grn_candidate_penalty_mask: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent

        if lambda_struct is None:
            lambda_struct = lam
        self.lambda_struct = float(lambda_struct)
        self.beta = float(beta)
        self.lam_dyn = lam_dyn
        self.normalized_regularization = normalized_regularization
        self.lr_positive = lr_positive
        self.omega_row_normalize = omega_row_normalize
        self.omega_norm_eps = float(omega_norm_eps)
        if not 0.0 <= float(tf_activity_dropout) < 1.0:
            raise ValueError("tf_activity_dropout must be in [0, 1).")
        self.tf_activity_dropout = float(tf_activity_dropout)
        self.lambda_omega_balance = float(lambda_omega_balance)
        self.omega_balance_tau = float(omega_balance_tau)
        self.omega_balance_mode = str(omega_balance_mode).lower()
        if self.omega_balance_mode not in {"none", "penalty", "hard"}:
            raise ValueError("omega_balance_mode must be one of {'none', 'penalty', 'hard'}.")
        self.omega_balance_iters = int(omega_balance_iters)
        if self.lambda_omega_balance < 0.0:
            raise ValueError("lambda_omega_balance must be nonnegative.")
        if self.omega_balance_tau <= 0.0:
            raise ValueError("omega_balance_tau must be positive.")
        if self.omega_balance_iters < 1:
            raise ValueError("omega_balance_iters must be >= 1.")
        self.omega_row_max_share = float(omega_row_max_share)
        if not 0.0 < self.omega_row_max_share <= 1.0:
            raise ValueError("omega_row_max_share must be in (0, 1].")
        self.grn_tf_load_alpha = float(grn_tf_load_alpha)
        if self.grn_tf_load_alpha < 0.0:
            raise ValueError("grn_tf_load_alpha must be nonnegative; use 0 to disable.")
        self.grn_prior_penalty_weight = float(grn_prior_penalty_weight)
        self.grn_candidate_penalty_weight = float(grn_candidate_penalty_weight)
        self.grn_disallowed_penalty_weight = float(grn_disallowed_penalty_weight)
        self.pathway_tf_penalty_weight = float(pathway_tf_penalty_weight)
        for name, value in [
            ("grn_prior_penalty_weight", self.grn_prior_penalty_weight),
            ("grn_candidate_penalty_weight", self.grn_candidate_penalty_weight),
            ("grn_disallowed_penalty_weight", self.grn_disallowed_penalty_weight),
            ("pathway_tf_penalty_weight", self.pathway_tf_penalty_weight),
        ]:
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        self.lr_degree_normalize = bool(lr_degree_normalize)
        self.lr_niche_contrast = bool(lr_niche_contrast)
        self.encoder_niche_mode = str(encoder_niche_mode).lower()
        if self.encoder_niche_mode not in {"none", "concat", "add"}:
            raise ValueError("encoder_niche_mode must be one of {'none', 'concat', 'add'}.")
        self.encoder_niche_mix_weight = float(encoder_niche_mix_weight)
        if not np.isfinite(self.encoder_niche_mix_weight):
            raise ValueError("encoder_niche_mix_weight must be finite.")
        self.niche_effect_mode = str(niche_effect_mode).lower()
        if self.niche_effect_mode not in {"cascade", "direct_mlp"}:
            raise ValueError("niche_effect_mode must be one of {'cascade', 'direct_mlp'}.")
        self.niche_residual_mode = str(niche_residual_mode).lower()
        if self.niche_residual_mode not in {"none", "target_mlp"}:
            raise ValueError("niche_residual_mode must be one of {'none', 'target_mlp'}.")
        self.disable_dynamic_grn = bool(disable_dynamic_grn)
        if likelihood not in {"mse", "log_mse"}:
            raise ValueError(
                "SPARC expects log-normalized expression input and "
                "supports only likelihood='mse' or the legacy alias 'log_mse'."
            )
        self.likelihood = likelihood

        # Backward-compatible aliases used by some external scripts.
        self.lam_grn = self.lambda_struct
        self.lam_lr = self.lambda_struct
        self.lam_omega = self.lambda_struct

        self.n_targets = int(sum(target_index))
        self.n_regulators = int(sum(regulator_index))
        self.regulator_index = regulator_index
        self.target_index = target_index

        if encoder_input_mask is None:
            encoder_input_mask = torch.ones(n_input, dtype=torch.float32)
        elif not torch.is_tensor(encoder_input_mask):
            encoder_input_mask = torch.tensor(encoder_input_mask, dtype=torch.float32)
        encoder_input_mask = encoder_input_mask.to(dtype=torch.float32).flatten()
        if encoder_input_mask.shape[0] != n_input:
            raise ValueError(
                f"encoder_input_mask must have length {n_input}, "
                f"got {encoder_input_mask.shape[0]}"
            )
        self.register_buffer("encoder_input_mask", encoder_input_mask)

        if lr_mask is not None:
            lr_tensor = torch.tensor(lr_mask, dtype=torch.float32)
            self.register_buffer("M_LR", lr_tensor)
            lig_idx, rec_idx = torch.where(lr_tensor > 0)
            self.register_buffer("lig_indices", lig_idx)
            self.register_buffer("rec_indices", rec_idx)
            self.register_buffer("lr_receptor_degree", lr_tensor.sum(dim=0).clamp_min(1.0))
        else:
            self.M_LR = None
            self.lr_receptor_degree = None

        if rec_tf_mask is not None:
            rec_tf_tensor = torch.tensor(rec_tf_mask, dtype=torch.float32)
            if rec_tf_tensor.shape == (n_input, n_input):
                regulator_mask = torch.tensor(regulator_index, dtype=torch.bool)
                rec_tf_tensor = rec_tf_tensor[:, regulator_mask]
            if rec_tf_tensor.shape != (n_input, self.n_regulators):
                raise ValueError(
                    "rec_tf_mask must have shape "
                    f"{(n_input, n_input)} or {(n_input, self.n_regulators)}, "
                    f"got {tuple(rec_tf_tensor.shape)}"
                )
            self.register_buffer("M_RTF", rec_tf_tensor)
        else:
            self.M_RTF = None

        if pathway_tf_mask is not None:
            pathway_tf_tensor = torch.tensor(pathway_tf_mask, dtype=torch.float32)
            if pathway_tf_tensor.shape == (n_input, n_input):
                regulator_mask = torch.tensor(regulator_index, dtype=torch.bool)
                pathway_tf_tensor = pathway_tf_tensor[:, regulator_mask]
            if pathway_tf_tensor.shape != (n_input, self.n_regulators):
                raise ValueError(
                    "pathway_tf_mask must have shape "
                    f"{(n_input, n_input)} or {(n_input, self.n_regulators)}, "
                    f"got {tuple(pathway_tf_tensor.shape)}"
                )
            self.register_buffer("M_PATHWAY_TF", pathway_tf_tensor)
        else:
            self.M_PATHWAY_TF = None

        self.register_buffer("M_GRN", skeleton)
        if grn_prior_mask is None:
            grn_prior_mask = skeleton
        if not torch.is_tensor(grn_prior_mask):
            grn_prior_mask = torch.tensor(grn_prior_mask, dtype=torch.float32)
        grn_prior_mask = grn_prior_mask.to(dtype=torch.float32)
        if tuple(grn_prior_mask.shape) != tuple(skeleton.shape):
            raise ValueError(
                "grn_prior_mask must have the same transformed shape as skeleton; "
                f"got {tuple(grn_prior_mask.shape)} vs {tuple(skeleton.shape)}"
            )
        grn_prior_mask = ((grn_prior_mask > 0) & (skeleton > 0)).to(dtype=torch.float32)
        grn_candidate_mask = ((skeleton > 0) & (grn_prior_mask <= 0)).to(dtype=torch.float32)
        self.register_buffer("M_GRN_PRIOR", grn_prior_mask)
        self.register_buffer("M_GRN_CANDIDATE", grn_candidate_mask)
        if grn_candidate_penalty_mask is None:
            grn_candidate_penalty_mask = grn_candidate_mask
        if not torch.is_tensor(grn_candidate_penalty_mask):
            grn_candidate_penalty_mask = torch.tensor(grn_candidate_penalty_mask, dtype=torch.float32)
        grn_candidate_penalty_mask = grn_candidate_penalty_mask.to(dtype=torch.float32)
        if tuple(grn_candidate_penalty_mask.shape) != tuple(skeleton.shape):
            raise ValueError(
                "grn_candidate_penalty_mask must have the same transformed shape as skeleton; "
                f"got {tuple(grn_candidate_penalty_mask.shape)} vs {tuple(skeleton.shape)}"
            )
        grn_candidate_penalty_mask = torch.clamp(grn_candidate_penalty_mask, min=0.0)
        grn_candidate_penalty_mask = grn_candidate_penalty_mask * grn_candidate_mask
        self.register_buffer("M_GRN_CANDIDATE_PENALTY", grn_candidate_penalty_mask)

        if fixed_lr_weights is not None:
            fixed_lr_tensor = torch.tensor(fixed_lr_weights, dtype=torch.float32)
            if fixed_lr_tensor.shape != (n_input, n_input):
                raise ValueError(
                    f"fixed_lr_weights must have shape {(n_input, n_input)}, "
                    f"got {tuple(fixed_lr_tensor.shape)}"
                )
            self.register_buffer("fixed_lr_weights", fixed_lr_tensor)
        else:
            self.fixed_lr_weights = None

        if fixed_omega_weights is not None:
            fixed_omega_tensor = torch.tensor(fixed_omega_weights, dtype=torch.float32)
            if fixed_omega_tensor.shape == (n_input, n_input):
                regulator_mask = torch.tensor(regulator_index, dtype=torch.bool)
                fixed_omega_tensor = fixed_omega_tensor[:, regulator_mask]
            if fixed_omega_tensor.shape != (n_input, self.n_regulators):
                raise ValueError(
                    "fixed_omega_weights must have shape "
                    f"{(n_input, n_input)} or {(n_input, self.n_regulators)}, "
                    f"got {tuple(fixed_omega_tensor.shape)}"
                )
            self.register_buffer("fixed_omega_weights", fixed_omega_tensor)
        else:
            self.fixed_omega_weights = None

        encoder_n_input = n_input * 2 if self.encoder_niche_mode == "concat" else n_input
        self.z_encoder_intra = Encoder(
            encoder_n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution="normal",
            use_batch_norm="both",
            use_layer_norm="both",
        )

        if self.niche_effect_mode == "direct_mlp":
            self.direct_niche_to_tf = nn.Sequential(
                nn.Linear(n_input, n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(n_hidden, self.n_regulators),
            )
        else:
            self.direct_niche_to_tf = None

        if self.niche_residual_mode == "target_mlp":
            self.niche_to_target_residual = nn.Sequential(
                nn.Linear(n_input, n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(n_hidden, self.n_targets),
            )
        else:
            self.niche_to_target_residual = None

        self.grn_dec_state = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.n_targets * self.n_regulators),
            nn.Tanh(),
        )

        self.W_global = nn.Parameter(torch.empty(self.n_targets, self.n_regulators))
        self.lambda_lr = nn.Parameter(torch.empty(n_input, n_input))
        self.omega = nn.Parameter(torch.empty(n_input, self.n_regulators))
        self.pathway_tf = nn.Parameter(torch.empty(n_input, self.n_regulators))

        self.target_bias = nn.Parameter(torch.zeros(self.n_targets))

        dynamic_scale_raw = _softplus_inverse(dynamic_scale_init)
        self.dynamic_scale_unconstr = nn.Parameter(torch.tensor(dynamic_scale_raw))

        self._reset_biophysical_parameters(
            lr_init=lr_init,
            lr_init_noise=lr_init_noise,
            communication_init=communication_init,
            w_init=w_init,
        )

    def _reset_biophysical_parameters(
        self,
        lr_init: float,
        lr_init_noise: float,
        communication_init: float,
        w_init: float,
    ) -> None:
        with torch.no_grad():
            self.W_global.normal_(0.0, w_init)
            self.W_global.mul_(self.M_GRN)

            if self.lr_positive:
                self.lambda_lr.fill_(_softplus_inverse(max(lr_init, 1e-8)))
                if lr_init_noise > 0:
                    self.lambda_lr.add_(torch.randn_like(self.lambda_lr) * lr_init_noise)
            else:
                self.lambda_lr.normal_(0.0, lr_init)

            if self.M_LR is not None:
                off_mask = self.M_LR <= 0
                if self.lr_positive:
                    self.lambda_lr[off_mask] = _softplus_inverse(1e-8)
                else:
                    self.lambda_lr[off_mask] = 0.0

            self.omega.normal_(0.0, communication_init)
            if self.M_RTF is not None:
                self.omega.mul_(self.M_RTF)

            self.pathway_tf.normal_(0.0, communication_init)
            if self.M_PATHWAY_TF is not None:
                self.pathway_tf.mul_(self.M_PATHWAY_TF)
            else:
                self.pathway_tf.zero_()

    def _effective_lr_weights(self):
        if self.fixed_lr_weights is not None:
            eff_lambda = self.fixed_lr_weights
        else:
            eff_lambda = F.softplus(self.lambda_lr) if self.lr_positive else self.lambda_lr
        if self.M_LR is not None:
            eff_lambda = eff_lambda * self.M_LR
        return eff_lambda

    def _effective_omega_weights(self):
        eff_omega = self.fixed_omega_weights if self.fixed_omega_weights is not None else self.omega
        if self.M_RTF is not None:
            eff_omega = eff_omega * self.M_RTF
        if self.omega_row_normalize:
            row_norm = eff_omega.abs().sum(dim=1, keepdim=True)
            eff_omega = torch.where(
                row_norm > self.omega_norm_eps,
                eff_omega / row_norm.clamp_min(self.omega_norm_eps),
                torch.zeros_like(eff_omega),
            )
        if self.omega_row_max_share < 1.0:
            eff_omega = self._apply_omega_row_max_cap(eff_omega)
        if self.omega_balance_mode == "hard" and self.fixed_omega_weights is None:
            eff_omega = self._apply_omega_column_cap(eff_omega)
        if self.omega_row_max_share < 1.0:
            eff_omega = self._apply_omega_row_max_cap(eff_omega)
        return eff_omega

    def _effective_pathway_tf_weights(self):
        if self.M_PATHWAY_TF is None:
            return None
        return self.pathway_tf * self.M_PATHWAY_TF

    def _effective_global_grn_weights(self):
        w = self.W_global * self.M_GRN
        if self.grn_tf_load_alpha <= 0.0 or w.shape[1] <= 1:
            return w

        col_loads = torch.sum(torch.abs(w), dim=0)
        supported = (torch.sum(self.M_GRN, dim=0) > 0) & (col_loads > self.omega_norm_eps)
        if not torch.any(supported):
            return w
        target_load = col_loads[supported].mean() * self.grn_tf_load_alpha
        scale_supported = torch.clamp(
            target_load / col_loads[supported].clamp_min(self.omega_norm_eps),
            max=1.0,
        )
        scale = torch.ones_like(col_loads)
        scale[supported] = scale_supported
        return w * scale.detach().unsqueeze(0)

    @staticmethod
    def _safe_active_count(mask: torch.Tensor) -> torch.Tensor:
        return mask.sum().clamp_min(1.0)

    def _regularizer_norm(self, values: torch.Tensor, mask=None) -> torch.Tensor:
        penalty = torch.sum(torch.abs(values if mask is None else values * mask))
        if not self.normalized_regularization:
            return penalty
        if mask is None:
            denom = torch.tensor(values.numel(), dtype=values.dtype, device=values.device)
        else:
            denom = self._safe_active_count(mask).to(dtype=values.dtype, device=values.device)
        return penalty / denom

    def _omega_prior_share(self, eff_omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.M_RTF is not None:
            support = (self.M_RTF > 0).to(dtype=eff_omega.dtype, device=eff_omega.device)
            prior_share = support.sum(dim=0)
            supported = prior_share > 0
            prior_share = prior_share / prior_share.sum().clamp_min(self.omega_norm_eps)
        else:
            supported = torch.ones(eff_omega.shape[1], dtype=torch.bool, device=eff_omega.device)
            prior_share = torch.full(
                (eff_omega.shape[1],),
                1.0 / max(1, eff_omega.shape[1]),
                dtype=eff_omega.dtype,
                device=eff_omega.device,
            )
        return prior_share, supported

    def _apply_omega_column_cap(self, eff_omega: torch.Tensor) -> torch.Tensor:
        if eff_omega.shape[1] <= 1:
            return eff_omega
        out = eff_omega
        prior_share, supported = self._omega_prior_share(out)
        cap = (self.omega_balance_tau * prior_share).clamp_min(self.omega_norm_eps)
        for _ in range(self.omega_balance_iters):
            abs_omega = out.abs()
            total_load = abs_omega.sum().clamp_min(self.omega_norm_eps)
            column_share = abs_omega.sum(dim=0) / total_load
            scale = torch.ones_like(column_share)
            over_cap = supported & (column_share > cap)
            scale = torch.where(over_cap, cap / column_share.clamp_min(self.omega_norm_eps), scale)
            out = out * scale.unsqueeze(0)
            row_norm = out.abs().sum(dim=1, keepdim=True)
            out = torch.where(
                row_norm > self.omega_norm_eps,
                out / row_norm.clamp_min(self.omega_norm_eps),
                torch.zeros_like(out),
            )
        return out

    def _apply_omega_row_max_cap(self, eff_omega: torch.Tensor) -> torch.Tensor:
        if eff_omega.shape[1] <= 1:
            return eff_omega
        out = eff_omega
        support = out.abs() > self.omega_norm_eps
        support_count = support.sum(dim=1, keepdim=True).clamp_min(1)
        feasible_cap = torch.maximum(
            torch.full_like(out[:, :1], self.omega_row_max_share),
            1.0 / support_count.to(dtype=out.dtype),
        )
        for _ in range(self.omega_balance_iters):
            abs_omega = out.abs()
            row_norm = abs_omega.sum(dim=1, keepdim=True)
            share = abs_omega / row_norm.clamp_min(self.omega_norm_eps)
            scale = torch.ones_like(out)
            over_cap = support & (share > feasible_cap)
            scale = torch.where(
                over_cap,
                feasible_cap / share.clamp_min(self.omega_norm_eps),
                scale,
            )
            out = out * scale
            row_norm = out.abs().sum(dim=1, keepdim=True)
            out = torch.where(
                row_norm > self.omega_norm_eps,
                out / row_norm.clamp_min(self.omega_norm_eps),
                torch.zeros_like(out),
            )
        return out

    def _omega_column_balance_loss(self, eff_omega: torch.Tensor) -> torch.Tensor:
        if (
            self.omega_balance_mode != "penalty"
            or self.lambda_omega_balance <= 0.0
            or eff_omega.shape[1] <= 1
        ):
            return torch.zeros((), dtype=eff_omega.dtype, device=eff_omega.device)

        abs_omega = eff_omega.abs()
        total_load = abs_omega.sum().clamp_min(self.omega_norm_eps)
        column_share = abs_omega.sum(dim=0) / total_load
        prior_share, supported = self._omega_prior_share(eff_omega)

        cap = (self.omega_balance_tau * prior_share).clamp_min(self.omega_norm_eps)
        relative_excess = F.relu(column_share - cap) / cap
        return self.lambda_omega_balance * torch.sum(relative_excess[supported] ** 2)

    def _get_inference_input(self, tensors):
        inps = {"x": tensors[REGISTRY_KEYS.X_KEY]}
        if REGISTRY_KEYS.X_NICHE_KEY in tensors:
            inps["x_niche"] = tensors[REGISTRY_KEYS.X_NICHE_KEY]
        return inps

    def _get_generative_input(self, tensors, inference_outputs):
        inps = {
            "x": tensors[REGISTRY_KEYS.X_KEY],
            "z_intra": inference_outputs["z_intra"],
        }
        if REGISTRY_KEYS.X_NICHE_KEY in tensors:
            inps["x_niche"] = tensors[REGISTRY_KEYS.X_NICHE_KEY]
        return inps

    def _decode_dynamic_weights(self, z_intra):
        n_obs = z_intra.shape[0]
        w_global = self._effective_global_grn_weights()
        w_base = w_global.unsqueeze(0).expand(n_obs, -1, -1)

        if self.disable_dynamic_grn:
            delta_w_state = torch.zeros_like(w_base)
            w_dynamic = w_base
            return w_dynamic, delta_w_state

        dynamic_scale = F.softplus(self.dynamic_scale_unconstr)

        delta_w_state = self.grn_dec_state(z_intra)
        delta_w_state = delta_w_state.view(n_obs, self.n_targets, self.n_regulators)
        delta_w_state = delta_w_state * dynamic_scale

        w_dynamic = (w_base + delta_w_state) * self.M_GRN.unsqueeze(0)

        return w_dynamic, delta_w_state

    def _compute_receptor_signal(self, y, y_niche=None):
        niche_input = y_niche if y_niche is not None else y
        if self.lr_niche_contrast and y_niche is not None:
            ligand_input = niche_input - y
        else:
            ligand_input = niche_input
        eff_lambda = self._effective_lr_weights()

        if self.M_LR is not None:
            ligands = ligand_input[:, self.lig_indices]
            lr_weights = eff_lambda[self.lig_indices, self.rec_indices]
            receptor_signal = torch.zeros_like(y)
            receptor_signal.scatter_add_(
                1,
                self.rec_indices.unsqueeze(0).expand(y.shape[0], -1),
                ligands * lr_weights,
            )
            if self.lr_degree_normalize:
                receptor_signal = receptor_signal / self.lr_receptor_degree.unsqueeze(0)
            receptor_signal = receptor_signal * y
        else:
            receptor_signal = torch.matmul(ligand_input, eff_lambda) * y

        return receptor_signal, eff_lambda

    def _apply_tf_activity_dropout(self, delta_a):
        if not self.training or self.tf_activity_dropout <= 0.0:
            return delta_a
        keep_prob = 1.0 - self.tf_activity_dropout
        # Drop whole TF modulation channels for the current minibatch. This is
        # intentionally stronger than elementwise dropout and discourages the
        # RTF layer from routing every receptor through one dominant TF.
        mask = torch.empty(
            1,
            delta_a.shape[1],
            dtype=delta_a.dtype,
            device=delta_a.device,
        ).bernoulli_(keep_prob)
        return delta_a * mask / keep_prob

    @auto_move_data
    def inference(self, x, x_niche=None, n_samples=1):
        y_encoder = x * self.encoder_input_mask.unsqueeze(0)
        if self.encoder_niche_mode == "add":
            if x_niche is None:
                niche_encoder = torch.zeros_like(x)
            else:
                niche_encoder = x_niche
            y_encoder = (x + self.encoder_niche_mix_weight * niche_encoder)
            y_encoder = y_encoder * self.encoder_input_mask.unsqueeze(0)
        elif self.encoder_niche_mode == "concat":
            if x_niche is None:
                niche_encoder = torch.zeros_like(x)
            else:
                niche_encoder = x_niche
            y_encoder = torch.cat([y_encoder, niche_encoder], dim=-1)
        qz_m_intra, qz_v_intra, z_intra = self.z_encoder_intra(y_encoder)

        if n_samples > 1:
            qz_m_intra = qz_m_intra.unsqueeze(0).expand(n_samples, -1, -1)
            qz_v_intra = qz_v_intra.unsqueeze(0).expand(n_samples, -1, -1)
            z_intra = Normal(qz_m_intra, qz_v_intra.sqrt()).sample()

        return {
            "z_intra": z_intra,
            "qz_m_intra": qz_m_intra,
            "qz_v_intra": qz_v_intra,
        }

    @auto_move_data
    def generative(self, x, z_intra, x_niche=None):
        y = x
        a_base = y[:, self.regulator_index]

        w_dynamic, delta_w_state = self._decode_dynamic_weights(z_intra)
        eff_omega = self._effective_omega_weights()
        if self.niche_effect_mode == "direct_mlp":
            niche_input = x_niche if x_niche is not None else y
            if self.lr_niche_contrast and x_niche is not None:
                niche_input = niche_input - y
            delta_a_lr_raw = self.direct_niche_to_tf(niche_input)
            receptor_signal = torch.zeros_like(y)
            eff_lambda = self._effective_lr_weights()
        else:
            receptor_signal, eff_lambda = self._compute_receptor_signal(y, x_niche)
            delta_a_lr_raw = torch.matmul(receptor_signal, eff_omega)
        eff_pathway_tf = self._effective_pathway_tf_weights()
        if eff_pathway_tf is None:
            delta_a_pathway = torch.zeros_like(delta_a_lr_raw)
        else:
            delta_a_pathway = torch.matmul(y, eff_pathway_tf)
        delta_a_raw = delta_a_lr_raw + delta_a_pathway
        delta_a = self._apply_tf_activity_dropout(delta_a_raw)
        a_tilde = torch.clamp(a_base + delta_a, min=0.0)

        target_logits = torch.bmm(w_dynamic, a_tilde.unsqueeze(-1)).squeeze(-1)
        target_logits = target_logits + self.target_bias
        if self.niche_residual_mode == "target_mlp":
            residual_input = x_niche if x_niche is not None else y
            niche_target_residual = self.niche_to_target_residual(residual_input)
            target_logits = target_logits + niche_target_residual
        else:
            niche_target_residual = torch.zeros_like(target_logits)
        x_hat = F.softplus(target_logits)

        return {
            "x_hat": x_hat,
            "f_intra": x_hat,
            "intra_logits": target_logits,
            "target_logits": target_logits,
            "W_dynamic": w_dynamic,
            "W_global_effective": self._effective_global_grn_weights(),
            "eff_lambda": eff_lambda,
            "eff_omega": eff_omega,
            "eff_pathway_tf": eff_pathway_tf,
            "delta_W_state": delta_w_state,
            "receptor_signal": receptor_signal,
            "delta_a_lr_raw": delta_a_lr_raw,
            "delta_a_pathway": delta_a_pathway,
            "delta_a_raw": delta_a_raw,
            "delta_a": delta_a,
            "a_tilde": a_tilde,
            "niche_target_residual": niche_target_residual,
        }

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight=1.0, n_obs=1.0):
        x = tensors[REGISTRY_KEYS.X_KEY]
        target_x = x[:, self.target_index]
        x_hat = generative_outputs["x_hat"]

        qz_m_intra = inference_outputs["qz_m_intra"]
        qz_v_intra = inference_outputs["qz_v_intra"]

        recon_x = F.mse_loss(
            x_hat,
            target_x,
            reduction="none",
        ).sum(dim=-1)

        kl_intra = kl(Normal(qz_m_intra, torch.sqrt(qz_v_intra)), Normal(0, 1)).sum(dim=1)

        # Confidence-asymmetric topological regularization. The forward graph
        # is restricted to M_GRN, while M_GRN_PRIOR marks strict database
        # edges. Edges in M_GRN_CANDIDATE are data-driven soft-prior candidates:
        # they can affect predictions but are penalized more strongly than
        # database-supported edges.
        mask_off_grn = 1 - self.M_GRN
        grn_loss = (
            self.lambda_struct
            * self.grn_disallowed_penalty_weight
            * self._regularizer_norm(self.W_global, mask_off_grn)
        )
        grn_loss += (
            self.lambda_struct
            * self.grn_prior_penalty_weight
            * self._regularizer_norm(self.W_global, self.M_GRN_PRIOR)
        )
        if torch.any(self.M_GRN_CANDIDATE > 0):
            grn_loss += (
                self.lambda_struct
                * self.grn_candidate_penalty_weight
                * self._regularizer_norm(
                    self.W_global * self.M_GRN_CANDIDATE_PENALTY,
                    self.M_GRN_CANDIDATE,
                )
            )

        eff_lambda = generative_outputs["eff_lambda"]
        direct_niche_loss = torch.zeros((), dtype=x.dtype, device=x.device)
        if self.niche_effect_mode == "direct_mlp":
            lr_loss = torch.zeros((), dtype=x.dtype, device=x.device)
            for param in self.direct_niche_to_tf.parameters():
                direct_niche_loss = direct_niche_loss + torch.mean(torch.abs(param))
            direct_niche_loss = direct_niche_loss * (self.lambda_struct * 0.1)
        elif self.M_LR is not None:
            lr_loss = (self.lambda_struct * 0.1) * self._regularizer_norm(
                eff_lambda, self.M_LR
            )
        else:
            lr_loss = self.lambda_struct * self._regularizer_norm(eff_lambda)

        if self.disable_dynamic_grn:
            dyn_loss = torch.zeros((), dtype=x.dtype, device=x.device)
            dynamic_scale_loss = torch.zeros((), dtype=x.dtype, device=x.device)
        else:
            dyn_mask = self.M_GRN.unsqueeze(0)
            dyn_raw = torch.sum(torch.abs(generative_outputs["delta_W_state"] * dyn_mask))
            if self.normalized_regularization:
                dyn_raw = dyn_raw / (max(1, x.shape[0]) * self._safe_active_count(self.M_GRN))
            else:
                dyn_raw = dyn_raw / max(1, x.shape[0])
            dyn_loss = self.lam_dyn * dyn_raw

            dynamic_scale_loss = self.lam_dyn * F.softplus(self.dynamic_scale_unconstr)

        if self.niche_effect_mode == "direct_mlp":
            omega_loss = torch.zeros((), dtype=x.dtype, device=x.device)
            omega_balance_loss = torch.zeros((), dtype=x.dtype, device=x.device)
        else:
            omega_mask = self.M_RTF if self.M_RTF is not None else None
            omega_values = self.fixed_omega_weights if self.fixed_omega_weights is not None else self.omega
            omega_loss = (self.lambda_struct * 0.1) * self._regularizer_norm(
                omega_values, omega_mask
            )
            omega_balance_loss = self._omega_column_balance_loss(generative_outputs["eff_omega"])
        if self.M_PATHWAY_TF is not None:
            pathway_tf_loss = (
                self.lambda_struct
                * self.pathway_tf_penalty_weight
                * self._regularizer_norm(self.pathway_tf, self.M_PATHWAY_TF)
            )
        else:
            pathway_tf_loss = torch.zeros((), dtype=x.dtype, device=x.device)

        niche_residual_loss = torch.zeros((), dtype=x.dtype, device=x.device)
        if self.niche_residual_mode == "target_mlp":
            for param in self.niche_to_target_residual.parameters():
                niche_residual_loss = niche_residual_loss + torch.mean(torch.abs(param))
            niche_residual_loss = niche_residual_loss * (self.lambda_struct * 0.1)

        local_loss = torch.mean(recon_x + (kl_intra * kl_weight * self.beta))
        total_loss = (
            local_loss
            + grn_loss
            + lr_loss
            + direct_niche_loss
            + dyn_loss
            + dynamic_scale_loss
            + omega_loss
            + omega_balance_loss
            + pathway_tf_loss
            + niche_residual_loss
        )

        return LossOutput(
            loss=total_loss,
            reconstruction_loss=recon_x,
            kl_local=kl_intra,
        )
