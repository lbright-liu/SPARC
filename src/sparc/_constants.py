"""Registry key constants for SPARC data management."""


class REGISTRY_KEYS:
    """Constants for AnnData registry keys."""

    X_KEY = "X"  # raw gene expression counts
    SPATIAL_KEY = "spatial"  # spatial coordinates
    CCC_SCORE_KEY = "ccc_score"
    X_NICHE_KEY = "x_niche"  # precomputed neighborhood log-expression
    CELL_TYPE_KEY = "cell_type"
    BATCH_KEY = "batch"
