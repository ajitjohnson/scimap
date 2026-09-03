import importlib

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture
def clustering_adata():
    values = np.array(
        [
            [1.0, 0.0, 1.0],
            [1.2, 0.1, 0.9],
            [0.8, 0.0, 1.1],
            [1.1, 0.2, 1.2],
            [7.0, 6.0, 7.0],
            [7.2, 5.8, 6.9],
            [6.8, 6.1, 7.1],
            [7.1, 6.2, 6.8],
        ]
    )
    adata = AnnData(
        values,
        obs=pd.DataFrame(
            {"phenotype": ["immune"] * 4 + ["tumor"] * 4},
            index=[f"cell-{index}" for index in range(len(values))],
        ),
        var=pd.DataFrame(index=["A", "B", "C"]),
    )
    adata.raw = adata.copy()
    return adata


def test_cluster_kmeans_supports_gene_subsets_and_raw_or_processed_data(
    clustering_adata,
):
    from scimap.tools.cluster import cluster

    result = cluster(
        clustering_adata,
        method="kmeans",
        subset_genes=["A", "C"],
        use_raw=False,
        k=2,
        random_state=4,
        label="processed_clusters",
        verbose=False,
    )

    assert result is clustering_adata
    assert set(result.obs["processed_clusters"].astype(str)) == {"0", "1"}


def test_cluster_kmeans_subclusters_selected_groups_and_can_collapse_others(
    clustering_adata,
):
    from scimap.tools.cluster import cluster

    result = cluster(
        clustering_adata,
        method="kmeans",
        sub_cluster=True,
        sub_cluster_group="immune",
        sub_cluster_column="phenotype",
        k=2,
        log=False,
        random_state=3,
        label="subclusters",
        verbose=False,
    )

    assert (
        result.obs.loc[result.obs["phenotype"] == "immune", "subclusters"]
        .str.startswith("immune-")
        .all()
    )
    assert (
        result.obs.loc[result.obs["phenotype"] == "tumor", "subclusters"] == "tumor"
    ).all()

    collapsed = cluster(
        clustering_adata.copy(),
        method="kmeans",
        sub_cluster=True,
        sub_cluster_group=["immune"],
        sub_cluster_column="phenotype",
        collapse_labels=True,
        k=2,
        random_state=3,
        label="collapsed_clusters",
        verbose=False,
    )

    assert (
        collapsed.obs.loc[collapsed.obs["phenotype"] == "tumor", "collapsed_clusters"]
        .isna()
        .all()
    )


def test_cluster_leiden_and_phenograph_dispatch_without_expensive_backends(
    clustering_adata, monkeypatch
):
    module = importlib.import_module("scimap.tools.cluster")
    neighbors_calls = []

    def fake_pca(subset):
        subset.obsm["X_pca"] = np.arange(subset.n_obs * 2).reshape(subset.n_obs, 2)

    def fake_neighbors(subset, n_neighbors, n_pcs):
        neighbors_calls.append((n_neighbors, n_pcs, subset.n_obs))

    def fake_leiden(subset, resolution, random_state):
        subset.obs["leiden"] = [str(index % 2) for index in range(subset.n_obs)]

    phenograph_calls = []

    def fake_phenograph(pca, k, primary_metric):
        phenograph_calls.append((pca.shape, k, primary_metric))
        return np.arange(len(pca)) % 2, None, None

    monkeypatch.setattr(module.sc.tl, "pca", fake_pca)
    monkeypatch.setattr(module.sc.pp, "neighbors", fake_neighbors)
    monkeypatch.setattr(module.sc.tl, "leiden", fake_leiden)
    monkeypatch.setattr(module.sce.tl, "phenograph", fake_phenograph)

    leiden = module.cluster(
        clustering_adata.copy(),
        method="leiden",
        sub_cluster=True,
        sub_cluster_group="tumor",
        sub_cluster_column="phenotype",
        use_raw=True,
        log=False,
        nearest_neighbors=99,
        n_pcs=2,
        label="leiden_subclusters",
        verbose=False,
    )
    phenograph = module.cluster(
        clustering_adata.copy(),
        method="phenograph",
        use_raw=False,
        nearest_neighbors=99,
        phenograph_clustering_metric="cosine",
        label="phenograph_clusters",
        verbose=False,
    )

    assert neighbors_calls == [(5, 2, 4)]
    assert (
        leiden.obs.loc[leiden.obs["phenotype"] == "tumor", "leiden_subclusters"]
        .str.startswith("tumor-")
        .all()
    )
    assert (
        leiden.obs.loc[leiden.obs["phenotype"] == "immune", "leiden_subclusters"]
        == "immune"
    ).all()
    assert phenograph_calls == [((8, 2), 5, "cosine")]
    assert set(phenograph.obs["phenograph_clusters"].astype(str)) == {"0", "1"}


def test_cluster_reads_and_writes_h5ad_paths(clustering_adata, tmp_path):
    from scimap.tools.cluster import cluster

    input_path = tmp_path / "input.h5ad"
    output_dir = tmp_path / "clustered"
    clustering_adata.write_h5ad(input_path)

    assert (
        cluster(
            input_path.as_posix(),
            method="kmeans",
            k=2,
            random_state=1,
            verbose=False,
            output_dir=output_dir,
        )
        is None
    )
    assert (output_dir / input_path.name).is_file()
