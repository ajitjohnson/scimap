import numpy as np
import pandas as pd
from anndata import AnnData

import scimap as sm


def test_spatial_lda_returns_normalized_sklearn_weights():
    adata = AnnData(np.ones((8, 2)))
    adata.obs = pd.DataFrame(
        {
            "imageid": ["image_1"] * 4 + ["image_2"] * 4,
            "phenotype": ["T cell", "B cell", "T cell", "Myeloid"] * 2,
            "X_centroid": [0, 1, 2, 3, 0, 1, 2, 3],
            "Y_centroid": [0, 0, 0, 0, 1, 1, 1, 1],
        },
        index=["cell_" + str(index) for index in range(8)],
    )

    result = sm.tl.spatial_lda(
        adata,
        phenotype="phenotype",
        num_motifs=2,
        method="knn",
        knn=2,
        random_state=0,
        max_iter=2,
        verbose=False,
    )

    motif_weights = result.uns["spatial_lda"]
    phenotype_probabilities = result.uns["spatial_lda_probability"]

    assert list(motif_weights.columns) == ["Motif_0", "Motif_1"]
    assert motif_weights.index.equals(adata.obs.index)
    assert np.allclose(motif_weights.sum(axis=1), 1)
    assert np.isfinite(motif_weights.to_numpy()).all()

    assert set(phenotype_probabilities.index) == {"T cell", "B cell", "Myeloid"}
    assert list(phenotype_probabilities.columns) == ["Motif_0", "Motif_1"]
    assert np.allclose(phenotype_probabilities.sum(axis=0), 1)
