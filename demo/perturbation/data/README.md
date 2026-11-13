# Processed Perturb-map input

- `perturb_map_processed.h5ad`: all 1,673 spots and 3,000 genes used to train
  the Perturb-map SPARC model. `X` contains library-size-normalized, `log1p`
  expression and `obsm["spatial"]` contains Visium coordinates.
- `perturb_map_priors.npz`: gene-aligned LR, receptor-TF, strict TF-target,
  and data-soft TF-target masks.
- `metadata.json`: dimensions, edge counts, and data provenance.

The raw Visium matrix and annotations are available from
[Perturb-map GSE193460](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460);
the included KP_1 slice is `GSM5808054`. The release file is already
preprocessed and can be passed directly to `demo/run_perturbation.py`.
