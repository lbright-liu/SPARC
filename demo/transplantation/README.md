# hColon virtual microenvironment transplantation

This workflow uses the complete processed hColon SPARC training input: 30,372
cells and all 425 measured genes. The 66 tumor-like fibroblasts used as held-out
experimental truth were excluded before model training, as in the paper.

Run a short installation check from the repository root:

```bash
python demo/run_transplantation.py --epochs 1 --accelerator gpu --devices 1
```

Run with the paper workflow's model settings:

```bash
python demo/run_transplantation.py \
  --epochs 150 --batch-size 256 \
  --accelerator gpu --devices 1
```

The runner rebuilds nonnegative expression from raw counts using
`normalize_total(10000)` followed by `log1p`, trains SPARC, and calls
`predict_microenvironment_transplant()`. The intrinsic state of each of the 40
normal-like donor fibroblasts is fixed while it is paired with each of 66
target-local profiles computed from the 64 nearest cells in the 512-cell
tumor-like recipient environment. This is the `target_local_env_frozen_z`
design used in the main hColon comparison. Use `--scenario mean_environment`
to run the original `mean_env_frozen_z` alternative.

The source data are the Xenium Human Colon Cancer Add-on FFPE release in the
[10x Genomics public dataset portal](https://www.10xgenomics.com/datasets).
`scripts/prepare_hcolon_data.py` records how the processed release input,
donor/environment labels, local niche profiles, and prior bundle were built.
