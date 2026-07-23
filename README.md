# SPARC

**Pathway-Constrained Spatial GRN Inference for Interpretable Counterfactual Perturbation Modeling**

SPARC is a generative framework for inferring global and cell-specific gene
regulatory networks (GRNs) from spatial transcriptomics data. It separates an
intrinsic cell-state representation from extrinsic microenvironmental effects.
Spatial signals are routed through a masked ligand-receptor, receptor-to-TF,
and TF-to-target cascade rather than entering an unconstrained environmental
latent variable.

This repository contains the SPARC model implementation and a compact
scMultiSim example focused on two inference interfaces:

- global TF-target GRN inference;
- cell-specific dynamic GRN inference.

<img width="2230" height="1228" alt="34511d5242b673b335d8cff94edf7238" src="https://github.com/user-attachments/assets/b3062f57-f1f2-444b-9901-9202f756fb3b" />



## Installation

SPARC was developed with Python 3.10.20, PyTorch 2.4.1, CUDA 12.1,
scvi-tools 1.2.0, and AnnData 0.11.4. A CUDA-capable GPU is recommended.

### Conda environment (recommended)

```bash
git clone https://github.com/lbright-liu/SPARC.git
cd SPARC
conda env create -f environment.yml
conda activate sparc
pip install -e . --no-deps
```

### Existing Python environment

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

The pinned requirements use the PyTorch CUDA 12.1 wheel. CPU execution is also
supported, although training is slower.

## Repository Structure

```text
SPARC/
|-- src/sparc/                  # Core SPARC model and generative module
|-- demo/
|   |-- data/                   # Compact scMultiSim spatial dataset and priors
|   |-- results/                # Verified global and dynamic GRN outputs
|   `-- run_demo.py             # End-to-end training and inference example
|-- scripts/
|   `-- prepare_demo_data.py    # Rebuild the compact subset from full benchmark data
|-- environment.yml             # Reproducible Conda environment
|-- requirements.txt            # Pinned Python dependencies
`-- pyproject.toml               # Installable package configuration
```

## Demo Dataset

The included demo is a fixed, domain-balanced subset of the scMultiSim
high-noise-prior benchmark used in the SPARC paper.

| Property | Demo setting |
| --- | ---: |
| Cells | 500 |
| Spatial domains | 5 (100 cells per domain) |
| Genes | 300 |
| Ligands / receptors | 30 / 30 |
| TFs / targets | 40 / 200 |
| TF-target prior edges | 500 |
| Simulated true TF-target edges | 200 |
| Simulated dynamic edges | 70 |

Expression values are library-size normalized and `log1p` transformed. The
subset uses sampling seed 17 and retains the complete gene panel and all prior
masks.

## Quick Start

Run the complete example on one GPU:

```bash
python demo/run_demo.py --accelerator gpu --devices 1
```

The default run uses 30 epochs, batch size 128, and random seed 17. To run on
CPU:

```bash
python demo/run_demo.py --accelerator cpu --devices 1
```

The script performs four steps:

1. loads the spatial expression matrix and L-R, receptor-TF, and TF-target priors;
2. constructs the default data-soft GRN support;
3. trains SPARC on the compact dataset;
4. calls the global and dynamic GRN inference interfaces.

The two public calls are:

```python
# pandas.DataFrame with shape (n_tfs, n_targets)
global_grn = model.get_global_grn()

# numpy.ndarray with shape (n_cells, n_tfs, n_targets)
dynamic_grn = model.get_dynamic_grn()
```

See [`demo/run_demo.py`](demo/run_demo.py) for the complete executable setup,
including prior-mask construction and model initialization.

## Output Files

The demo writes the following files to `demo/results/`:

| File | Contents |
| --- | --- |
| `global_grn.csv` | Signed global TF-by-target regulatory weight matrix |
| `top_global_edges.csv` | Top 100 global edges ranked by absolute weight |
| `dynamic_grn.npz` | Cell-by-TF-by-target weights with cell and gene labels |
| `dynamic_grn_edge_summary.csv` | Per-edge mean, standard deviation, minimum, and maximum across cells |
| `run_summary.json` | Data dimensions, runtime, device, versions, and output shapes |

The compressed dynamic file contains four arrays:

```python
import numpy as np

result = np.load("demo/results/dynamic_grn.npz")
weights = result["weights"]          # cells x TFs x targets
cell_names = result["cell_names"]
tf_genes = result["tf_genes"]
target_genes = result["target_genes"]
```

## Verified Demo Run

The committed example outputs were generated with the command:

```bash
python demo/run_demo.py \
  --epochs 30 --batch-size 128 \
  --accelerator gpu --devices 1
```

| Item | Verified value |
| --- | --- |
| Hardware | NVIDIA GeForce RTX 3090 |
| Training time | ~10s |
| Data-soft candidate edges | 260 |
| Total modeled TF-target edges | 760 |
| Global GRN shape | 40 x 200 |
| Dynamic GRN shape | 500 x 40 x 200 |

## Input Conventions

For a new dataset, SPARC expects:

- an `AnnData` object containing log-normalized expression in `X` or a named layer;
- spatial coordinates in `adata.obsm["spatial"]`;
- Boolean gene masks identifying TFs and target genes;
- a target-by-TF GRN support mask;
- a gene-by-gene ligand-to-receptor mask;
- a gene-by-TF receptor-to-TF mask.

The prior knowledge masks for the ligand-receptor-TF-target cascade can be derived from an integrated, large-scale multi-layer signaling network resource (Nature Computational Science, 2026; GitHub repository: https://github.com/SunXQlab/CCCvelo).

Expression and spatial data are registered before model construction:

```python
from sparc import SPARC

SPARC.setup_anndata(
    adata,
    x_layer=None,          # or the name of a log-normalized expression layer
    spatial_key="spatial",
    sigma=0.18,
    n_neighbors=12,
)
```

The demo script provides a concrete reference for converting named genes and
edge lists into the required masks. Input priors must be aligned to
`adata.var_names`; TF and target order determines the corresponding output
axes.

## Reproducibility

The demo fixes Python, NumPy, PyTorch, CUDA, and scvi-tools versions through
`environment.yml` and `requirements.txt`. Python, NumPy, PyTorch, CUDA, and
scvi random states are initialized with seed 17, and deterministic PyTorch
operations are requested where available. Exact demo dimensions and software
versions are recorded in `demo/results/run_summary.json`.

## Citation

If you use SPARC, please cite the associated paper:

> Liliang Liu et al. SPARC: Pathway-Constrained Spatial GRN Inference for
> Interpretable Counterfactual Perturbation Modeling.

The bibliographic record will be updated after publication. Software citation
metadata are provided in [`CITATION.cff`](CITATION.cff).
