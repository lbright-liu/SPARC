# Perturb-map in-silico knockout

This workflow uses the complete processed Perturb-map KP_1 model input from
the SPARC experiment: 1,673 non-KO training spots and the full 3,000-gene
panel. It is not a downsampled cell subset.

Run a short installation check from the repository root:

```bash
python demo/run_perturbation.py --epochs 1 --accelerator gpu --devices 1
```

Run with the paper workflow's training and rollout settings:

```bash
python demo/run_perturbation.py \
  --epochs 150 --batch-size 128 \
  --genes Tgfbr2 Jak2 \
  --n-steps 30 --damping 0.25 \
  --accelerator gpu --devices 1
```

The runner trains only on non-KO tissue. It applies each knockout to the 208
unperturbed KP control spots, performs the fixed-point spatial rollout, and
writes a labeled spot-by-target delta matrix plus a gene-ranked summary.

The processed input is derived from Perturb-map accession `GSM5808054`.
`scripts/prepare_perturb_map_data.py` records how the release file and aligned
prior bundle were generated from the full SPARC model input.
