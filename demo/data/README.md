# Demo data

The files in this directory form a compact, domain-balanced subset of the
scMultiSim high-noise-prior benchmark used in the SPARC paper.

- `scmultisim_demo.h5ad` contains 500 cells (100 from each of five spatial
  domains) and the complete 300-gene panel.
- `scmultisim_demo_gold.npz` contains the ligand-receptor, receptor-TF, and
  TF-target masks and simulated GRN metadata required by the demo.
- `metadata.json` records the sampling seed, dimensions, and edge counts.

The subset is intended only to demonstrate the SPARC Python interfaces. The
reported paper benchmarks use all 3,000 cells, 150 training epochs, and three
model seeds.
