# SPARC

**Pathway-Constrained Spatial GRN Inference for Interpretable Counterfactual Perturbation Modeling**

SPARC learns global and cell-specific gene regulatory networks (GRNs) from
spatial transcriptomics. Its architecture separates intrinsic cell state from
extrinsic niche effects and routes spatial information through a masked
ligand-receptor, receptor-to-TF, and TF-to-target cascade.

This repository exposes three end-to-end workflows:

1. global and cell-specific GRN inference;
2. in-silico gene perturbation with spatial rollout;
3. virtual microenvironment transplantation with a fixed receiver state.

<img width="2230" height="1228" alt="SPARC overview" src="https://github.com/user-attachments/assets/b9e37c5c-efe5-4116-8fed-7542cc855c86" />

## Installation

SPARC was developed with Python 3.10.20, PyTorch 2.4.1, CUDA 12.1,
scvi-tools 1.2.0, and AnnData 0.11.4. A CUDA-capable GPU is recommended.

```bash
git clone https://github.com/lbright-liu/SPARC.git
cd SPARC
conda env create -f environment.yml
conda activate sparc
pip install -e . --no-deps
```

To install in an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## Repository structure

```text
SPARC/
|-- src/sparc/                       # Model, generative module, prior builder
|-- resources/priors/                # LR, receptor-TF, and TF-target tables
|-- demo/
|   |-- data/                        # 500-cell scMultiSim GRN demo
|   |-- perturbation/data/           # Complete processed Perturb-map input
|   |-- transplantation/data/        # Complete processed hColon training input
|   |-- run_demo.py                  # Global and dynamic GRN inference
|   |-- run_perturbation.py          # In-silico knockout
|   `-- run_transplantation.py       # Virtual niche transplantation
`-- scripts/                         # Scripts that rebuild release data files
```

## 1. GRN inference

The included scMultiSim example is a fixed, domain-balanced 500-cell subset
of the paper benchmark. It retains the complete 300-gene panel and all prior
masks: 30 ligands, 30 receptors, 40 TFs, 200 targets, and 70 dynamic edges.
The subset is an API demonstration; the paper benchmark used 3,000 cells, 150
epochs, and three model seeds.

```bash
python demo/run_demo.py --accelerator gpu --devices 1
```

The public inference calls are:

```python
# pandas.DataFrame: TF x target
global_grn = model.get_global_grn()

# numpy.ndarray: cell x TF x target
dynamic_grn = model.get_dynamic_grn()
```

The runner writes `global_grn.csv`, `top_global_edges.csv`, a labeled
`dynamic_grn.npz`, an edge summary, and `run_summary.json` under
`demo/results/`.

## 2. In-silico perturbation

The Perturb-map release input contains all 1,673 non-KO training spots and the
full 3,000-gene panel used by SPARC; it is not a cell subset. Expression in
`X` was normalized to a library size of 10,000 and `log1p` transformed.

Installation check:

```bash
python demo/run_perturbation.py \
  --epochs 1 --genes Tgfbr2 \
  --accelerator gpu --devices 1
```

Recommended run with the original workflow settings:

```bash
python demo/run_perturbation.py \
  --epochs 150 --batch-size 128 \
  --genes Tgfbr2 Jak2 \
  --n-steps 30 --damping 0.25 \
  --accelerator gpu --devices 1
```

The underlying API is:

```python
delta = model.predict_perturbation(
    adata=adata,
    gene="Tgfbr2",
    ko_value=0.0,
    cell_mask=control_mask,
    n_steps=30,
    damping=0.25,
    return_dataframe=True,
)
# pandas.DataFrame: all cells x modeled target genes
```

The intervention is reclamped at each step while target residuals are updated
with a damped fixed-point rollout. Spatial feedback expands by one graph hop
per step. The runner exports a labeled control-spot-by-target delta matrix and
a gene response ranking. See
[`demo/perturbation/README.md`](demo/perturbation/README.md) for details.

## 3. Virtual microenvironment transplantation

The hColon release input contains all 30,372 cells and 425 genes used for
model training. The 66 tumor-like fibroblasts reserved as experimental truth
were excluded before this training input was created. The runner explicitly
rebuilds nonnegative expression from `layers["counts"]` using
`normalize_total(10000)` followed by `log1p`; upstream scaled z-scores must not
be used as SPARC input.

Installation check:

```bash
python demo/run_transplantation.py \
  --epochs 1 --accelerator gpu --devices 1
```

Recommended run with the original workflow settings:

```bash
python demo/run_transplantation.py \
  --epochs 150 --batch-size 256 \
  --accelerator gpu --devices 1
```

The public transplantation call is:

```python
result = model.predict_microenvironment_transplant(
    adata=adata,
    donor_mask=adata.obs["transplant_role"].eq("donor").to_numpy(),
    x_niche=target_local_profiles,
    cartesian_niche=True,
    return_dataframe=True,
)

baseline = result["baseline_df"]
transplanted = result["transplanted_df"]
delta = result["delta_df"]
```

The donor expression and intrinsic latent state remain fixed; only the niche
input is replaced. The default runner pairs every donor with every included
target-local profile, matching the `target_local_env_frozen_z` design used in
the hColon comparison. Passing `environment_mask` instead uses the mean
recipient profile (`mean_env_frozen_z`). See
[`demo/transplantation/README.md`](demo/transplantation/README.md) for details.

## Using a new spatial dataset

SPARC expects:

- nonnegative, log-normalized expression in `adata.X` or a named layer;
- spatial coordinates in `adata.obsm["spatial"]`;
- gene-aligned LR, receptor-TF, and TF-target support masks.

The bundled pathway tables can be intersected with any gene panel:

```python
from sparc import SPARC, build_cascade_priors

priors = build_cascade_priors(adata, "resources/priors")

SPARC.setup_anndata(
    adata,
    x_layer=None,
    spatial_key="spatial",
    sigma=80.0,       # choose for the coordinate scale of the dataset
    n_neighbors=15,
)

model = SPARC(
    adata,
    skeleton=priors["skeleton"],
    regulator_index=priors["regulator_index"],
    target_index=priors["target_index"],
    lr_mask=priors["lr_mask"],
    rec_tf_mask=priors["rec_tf_mask"],
    loss_preset="balanced",
    likelihood="mse",
)
model.initialize_grn_from_expression(scale=0.05)
model.train(max_epochs=150, batch_size=256, accelerator="gpu", devices=1)
```

All masks follow `adata.var_names`. TF and target order determines the
corresponding GRN output axes. The bundled prior resource is derived from the
integrated multi-layer signaling network distributed with
[CCCvelo](https://github.com/SunXQlab/CCCvelo) and its associated Nature
Computational Science 2026 publication.

## Data provenance

| Workflow | Included processed input | Original source |
| --- | --- | --- |
| GRN inference | 500-cell scMultiSim subset | scMultiSim-backed benchmark described in the paper |
| Perturbation | 1,673 spots x 3,000 genes | [Perturb-map GSE193460](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460), KP_1 slice `GSM5808054` |
| Transplantation | 30,372 cells x 425 genes | [10x Genomics public datasets](https://www.10xgenomics.com/datasets), Xenium Human Colon Cancer Add-on FFPE |

The preparation scripts under `scripts/` document how the processed release
files and gene-aligned prior bundles were generated. They do not need to be
run before using the included data.

## Citation

SPARC was accepted at KDD 2027. If you use this software or its processed
examples, please cite:

> Liliang Liu et al. SPARC: Pathway-Constrained Spatial GRN Inference for
> Interpretable Counterfactual Perturbation Modeling. KDD 2027.

Software citation metadata are provided in [`CITATION.cff`](CITATION.cff).
