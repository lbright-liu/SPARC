# Verified demo outputs

These files were generated from the included 500-cell scMultiSim subset with
SPARC 0.4.0, seed 17, 30 training epochs, and batch size 128:

```bash
python demo/run_demo.py --accelerator gpu --devices 1
```

The run used an NVIDIA GeForce RTX 3090 and completed the training call in
4.14 seconds. `global_grn.csv` has 40 TF rows and 200 target columns;
`dynamic_grn.npz` contains a `500 x 40 x 200` cell-specific weight tensor plus
the corresponding cell, TF, and target labels.

These outputs verify installation and interface behavior. They are not the
paper benchmark results because the demo uses a subset of cells and fewer
training epochs.
