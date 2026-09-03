import importlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture
def tiny_adata():
    values = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 5.0],
            [8.0, 7.0, 6.0],
            [9.0, 8.0, 7.0],
            [10.0, 9.0, 8.0],
        ]
    )
    obs = pd.DataFrame(index=[f"cell-{i}" for i in range(len(values))])
    var = pd.DataFrame(index=["A", "B", "C"])
    adata = AnnData(values.copy(), obs=obs, var=var)
    adata.raw = adata.copy()
    adata.layers["alternate"] = values + 100
    return adata


@pytest.mark.parametrize("layer", ["raw", None, "alternate"])
def test_ngraph_supports_every_data_source(tiny_adata, layer):
    from scimap.preprocessing.ngraph import nGraph

    result = nGraph(tiny_adata, layer=layer, k_neighbors=2)

    assert result is tiny_adata
    assert result.obsp["connectivities"].shape == (6, 6)
    np.testing.assert_array_equal(
        np.asarray(result.obsp["connectivities"].sum(axis=1)).ravel(),
        np.full(6, 2),
    )


def test_ngraph_runs_scaling_and_pca(tiny_adata):
    from scimap.preprocessing.ngraph import nGraph

    result = nGraph(
        tiny_adata,
        layer=None,
        standardScale=True,
        runPCA=True,
        k_neighbors=1,
    )

    assert result.obsp["connectivities"].nnz == 6


def test_log1p_overwrites_existing_layer_and_persists_path(
    tiny_adata, tmp_path, capsys
):
    from scimap.preprocessing.log1p import log1p

    path = tmp_path / "input.h5ad"
    tiny_adata.layers["log"] = np.zeros(tiny_adata.shape)
    tiny_adata.write_h5ad(path)

    assert log1p(str(path), layer="log", verbose=True) is None

    saved = ad.read_h5ad(path)
    np.testing.assert_allclose(saved.layers["log"], np.log1p(saved.raw.X))
    output = capsys.readouterr().out
    assert "already exists" in output
    assert "has been saved" in output


def test_log1p_rejects_missing_path_and_raw_data(tmp_path, tiny_adata):
    from scimap.preprocessing.log1p import log1p

    with pytest.raises(FileNotFoundError):
        log1p(str(tmp_path / "missing.h5ad"))

    tiny_adata.raw = None
    with pytest.raises(AttributeError, match="adata.raw does not exist"):
        log1p(tiny_adata)


def test_mcmicro_import_options(tmp_path):
    from scimap.preprocessing.mcmicro_to_scimap import mcmicro_to_scimap

    first = pd.DataFrame(
        {
            "DNA1": [1.0, 2.0, 3.0],
            "CD3_suffix": [4.0, np.inf, 6.0],
            "remove_suffix": [7.0, 8.0, 9.0],
            "X_centroid": [10, 11, 12],
            "Y_centroid": [20, 21, 22],
            "CellID": [1, 2, 3],
        }
    )
    second = first.iloc[:1].copy()
    first_path = tmp_path / "image-one.csv"
    second_path = tmp_path / "image-two.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)

    result = mcmicro_to_scimap(
        [first_path, second_path],
        remove_string_from_name="_suffix",
        drop_markers="remove",
        min_cells=2,
        log=False,
        verbose=False,
    )

    assert result.shape == (3, 1)
    assert result.var_names.tolist() == ["CD3"]
    assert result.obs["imageid"].unique().tolist() == ["image-one"]
    assert np.isfinite(result.X).all()
    np.testing.assert_allclose(result.layers["log"], np.log1p(result.X))


def test_mcmicro_main_forwards_parsed_arguments(monkeypatch):
    module = importlib.import_module("scimap.preprocessing.mcmicro_to_scimap")

    calls = []
    monkeypatch.setattr(
        module, "mcmicro_to_scimap", lambda **kwargs: calls.append(kwargs)
    )

    module.main(
        [
            "program",
            "--feature_table_path",
            "one.csv",
            "two.csv",
            "--random_sample",
            "4",
        ]
    )

    assert calls[0]["feature_table_path"] == ["one.csv", "two.csv"]
    assert calls[0]["random_sample"] == 4
