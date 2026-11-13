# Pathway prior tables

SPARC uses three integrated signaling-network tables:

| File | Edge type | Required columns |
| --- | --- | --- |
| `LigRecDB.csv` | ligand to receptor | `source`, `target` |
| `RecTFDB.csv` | receptor to transcription factor | `source`, `target`, `score` |
| `TFTGDB.csv` | transcription factor to target gene | `source`, `target`, `score` |

These tables are the prior resource used in the SPARC experiments and were
derived from the integrated multi-layer signaling network distributed with
[CCCvelo](https://github.com/SunXQlab/CCCvelo), associated with the Nature
Computational Science 2026 publication cited by the SPARC paper. Retain that
source attribution when redistributing derived masks or tables.

Build masks aligned to an AnnData gene panel with:

```python
from sparc import build_cascade_priors

priors = build_cascade_priors(adata, "resources/priors")
```

Database membership defines allowed pathway support; SPARC learns the edge
weights from the spatial expression data.
