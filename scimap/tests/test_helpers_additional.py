import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture
def helper_adata():
    obs = pd.DataFrame(
        {
            "imageid": ["one", "one", "two", "two"],
            "phenotype": ["keep", "drop", "keep", "drop"],
            "metadata": [1, 2, 3, 4],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    var = pd.DataFrame(index=["A", "B", "C"])
    adata = AnnData(np.arange(12, dtype=float).reshape(4, 3), obs=obs, var=var)
    adata.raw = adata.copy()
    adata.layers["alternate"] = adata.X + 20
    return adata


def test_drop_features_exercises_all_drop_modes(helper_adata):
    from scimap.helpers.dropFeatures import dropFeatures

    result = dropFeatures(
        helper_adata,
        drop_markers="B",
        drop_meta_columns=["metadata", "does-not-exist"],
        verbose=False,
    )

    assert result.obs_names.tolist() == ["c1", "c2", "c3", "c4"]
    assert result.var_names.tolist() == ["A", "C"]
    assert result.obs.columns.tolist() == ["imageid", "phenotype"]
    assert result.raw.shape == result.shape

    cells_dropped = dropFeatures(
        helper_adata.copy(), drop_cells="c1", subset_raw=False, verbose=False
    )
    assert cells_dropped.obs_names.tolist() == ["c2", "c3", "c4"]

    groups_dropped = dropFeatures(
        helper_adata.copy(),
        drop_groups="drop",
        groups_column=["phenotype"],
        subset_raw=False,
        verbose=False,
    )
    assert groups_dropped.obs_names.tolist() == ["c1", "c3"]


def test_drop_features_can_leave_raw_unmodified_and_validates_group_column(
    helper_adata,
):
    from scimap.helpers.dropFeatures import dropFeatures

    result = dropFeatures(helper_adata, drop_markers="A", subset_raw=False)
    assert result.shape == (4, 2)
    assert result.raw.shape == (4, 3)

    with pytest.raises(AssertionError, match="missing not found"):
        dropFeatures(helper_adata, drop_groups="drop", groups_column="missing")


def test_merge_adata_obs_combines_metadata_and_uses_richest_uns(helper_adata):
    from scimap.helpers.merge_adata_obs import merge_adata_obs

    first = helper_adata.copy()
    first.obs = first.obs[["imageid"]]
    first.uns["one"] = 1
    second = helper_adata.copy()
    second.obs = second.obs[["phenotype"]]
    second.uns.update({"one": 1, "two": 2})
    second.X = np.full(second.shape, 99.0)

    result = merge_adata_obs([first, second], verbose=False)

    assert result.obs.columns.tolist() == ["imageid", "phenotype"]
    np.testing.assert_array_equal(result.X, np.full(second.shape, 99.0))
    assert result.uns == second.uns


def test_merge_adata_obs_supports_paths_and_output_directory(helper_adata, tmp_path):
    from scimap.helpers.merge_adata_obs import merge_adata_obs

    first = helper_adata.copy()
    first.obs = first.obs[["imageid"]]
    second = helper_adata.copy()
    second.obs = second.obs[["phenotype"]]
    second.uns["extra"] = True
    first_path = tmp_path / "first.h5ad"
    second_path = tmp_path / "second.h5ad"
    output_dir = tmp_path / "nested" / "output"
    first.write_h5ad(first_path)
    second.write_h5ad(second_path)

    assert (
        merge_adata_obs([str(first_path), str(second_path)], output_dir=output_dir)
        is None
    )

    saved = ad.read_h5ad(output_dir / "combined_adata.h5ad")
    assert saved.obs.columns.tolist() == ["imageid", "phenotype"]
    assert saved.uns["extra"]


def test_merge_main_forwards_arguments(monkeypatch):
    module = importlib.import_module("scimap.helpers.merge_adata_obs")

    calls = []
    monkeypatch.setattr(
        module, "merge_adata_obs", lambda **kwargs: calls.append(kwargs)
    )

    module.main(["program", "--adata", "one.h5ad", "two.h5ad", "--output_dir", "out"])

    assert calls == [{"adata": ["one.h5ad", "two.h5ad"], "output_dir": "out"}]


@pytest.mark.parametrize(
    ("layer", "expected"),
    [
        ("raw", lambda adata: adata.raw.X),
        (None, lambda adata: adata.X),
        ("alternate", lambda adata: adata.layers["alternate"]),
    ],
)
def test_scimap_to_csv_supports_all_expression_sources(helper_adata, layer, expected):
    from scimap.helpers.scimap_to_csv import scimap_to_csv

    result = scimap_to_csv(helper_adata, layer=layer)

    assert result.columns[0] == "CellID"
    assert result["CellID"].tolist() == helper_adata.obs_names.tolist()
    np.testing.assert_array_equal(result[["A", "B", "C"]], expected(helper_adata))


def test_scimap_to_csv_reads_a_path_and_writes_named_file(helper_adata, tmp_path):
    from scimap.helpers.scimap_to_csv import scimap_to_csv

    input_path = tmp_path / "input.h5ad"
    output_dir = tmp_path / "exports"
    helper_adata.write_h5ad(input_path)

    assert (
        scimap_to_csv(str(input_path), output_dir=output_dir, file_name="cells") is None
    )
    exported = pd.read_csv(output_dir / "cells.csv")
    assert exported.shape == (4, 7)


def test_scimap_to_csv_main_forwards_arguments(monkeypatch):
    module = importlib.import_module("scimap.helpers.scimap_to_csv")

    calls = []
    monkeypatch.setattr(module, "scimap_to_csv", lambda **kwargs: calls.append(kwargs))

    module.main(
        [
            "program",
            "--adata",
            "input.h5ad",
            "--layer",
            "alternate",
            "--file_name",
            "cells",
        ]
    )

    assert calls[0]["adata"] == "input.h5ad"
    assert calls[0]["layer"] == "alternate"
    assert calls[0]["file_name"] == "cells"


class _Progress:
    def __init__(self, **kwargs):
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, value):
        self.updates.append(value)


def test_download_demo_data_downloads_all_chunks(monkeypatch, tmp_path):
    module = importlib.import_module("scimap.helpers.downloadDemoData")

    metadata = SimpleNamespace(
        status_code=200,
        json=lambda: {
            "files": [{"key": "demo.bin", "links": {"self": "https://files/demo.bin"}}]
        },
    )
    payload = SimpleNamespace(
        headers={"content-length": "6"},
        iter_content=lambda block_size: iter([b"abc", b"def"]),
    )
    responses = iter([metadata, payload])
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(module, "tqdm", _Progress)

    module.downloadDemoData(str(tmp_path / "downloads"), api_url="https://api/record")

    assert (tmp_path / "downloads" / "demo.bin").read_bytes() == b"abcdef"


def test_download_demo_data_handles_failed_record_request(
    monkeypatch, tmp_path, capsys
):
    module = importlib.import_module("scimap.helpers.downloadDemoData")

    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=503),
    )

    assert module.downloadDemoData(str(tmp_path), api_url="https://api/record") is None
    assert "HTTP status code: 503" in capsys.readouterr().out


def test_add_roi_omero_assigns_polygons_and_respects_subset():
    from scimap.helpers.addROI_omero import addROI_omero

    obs = pd.DataFrame(
        {
            "X_centroid": [10.0, 40.0, 10.0],
            "Y_centroid": [10.0, 40.0, 10.0],
            "imageid": ["one", "one", "two"],
        },
        index=["inside-one", "outside", "inside-two"],
    )
    adata = AnnData(np.zeros((3, 1)), obs=obs, var=pd.DataFrame(index=["A"]))
    adata.obs["region"] = "existing"
    roi = pd.DataFrame(
        {
            "Id": [1],
            "Name": ["Tumor"],
            "type": ["Rectangle"],
            "all_points": ["00,00 20,00 20,20 00,20"],
        }
    )

    result = addROI_omero(
        adata,
        roi,
        subset="one",
        overwrite=False,
        label="region",
        n_jobs=1,
    )

    assert result.obs.loc["inside-one", "region"] == "Tumor"
    assert result.obs.loc["outside", "region"] == "existing"
    assert result.obs.loc["inside-two", "region"] == "existing"


def test_animate_validates_inputs_and_builds_animation(monkeypatch, tmp_path):
    module = importlib.import_module("scimap.helpers.animate")

    obs = pd.DataFrame(
        {
            "X_centroid": [0.0, 1.0, 2.0, 3.0],
            "Y_centroid": [0.0, 2.0, 1.0, 3.0],
            "imageid": ["one", "one", "two", "two"],
            "phenotype": pd.Categorical(["A", "A", "B", "B"]),
        },
        index=["c1", "c2", "c3", "c4"],
    )
    adata = AnnData(
        np.arange(8, dtype=float).reshape(4, 2),
        obs=obs,
        var=pd.DataFrame(index=["M1", "M2"]),
    )
    adata.raw = adata.copy()
    adata.layers["scaled"] = adata.X / 2
    adata.obsm["umap"] = np.array([[0, 0], [1, 2], [2, 1], [3, 3]], dtype=float)

    animation = Mock()
    monkeypatch.setattr(module, "FuncAnimation", Mock(return_value=animation))
    monkeypatch.setattr(module.plt, "show", Mock(return_value="shown"))

    result = module.animate(
        adata,
        color="phenotype",
        subset="one",
        subsample=1.0,
        n_frames=20,
        final_frame=2,
        reverse=False,
        plot_legend=True,
        title=True,
        save_animation=str(tmp_path / "transition"),
        verbose=False,
    )

    assert result == "shown"
    module.FuncAnimation.assert_called_once()
    animation.save.assert_called_once_with(
        str(tmp_path / "transition") + "_scimap.gif",
        writer="imagemagick",
        fps=24,
    )

    with pytest.raises(ValueError, match="Only a single value"):
        module.animate(adata, color=["phenotype", "M1"])

    missing_embedding = adata.copy()
    del missing_embedding.obsm["umap"]
    with pytest.raises(KeyError, match="Please run"):
        module.animate(missing_embedding, color="phenotype")
