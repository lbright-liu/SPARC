# Processed hColon input

- `hcolon_processed_counts.h5ad`: all 30,372 training cells and 425 genes,
  with raw counts in `layers["counts"]`, coordinates in `obsm["spatial"]`, and
  donor/environment membership in `obs["transplant_role"]`.
- `hcolon_priors.npz`: gene-aligned LR, receptor-TF, strict TF-target, and
  data-soft TF-target masks.
- `hcolon_target_local_niches.npz`: 66 local recipient-niche profiles, each
  averaging the 64 nearest cells around one held-out tumor-like fibroblast.
- `metadata.json`: dimensions, edge counts, and data provenance.

The runner intentionally reconstructs log-normalized expression from counts.
The scaled z-score matrix created by upstream annotation is not a valid SPARC
input because SPARC's decoder operates on nonnegative expression.
