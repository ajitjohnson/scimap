import importlib
from unittest.mock import Mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.spatial import Voronoi


def _adata():
    obs = pd.DataFrame(
        {
            "imageid": ["image-1"] * 6,
            "cluster": ["a", "a", "a", "b", "b", "b"],
            "X_centroid": [0, 1, 2, 3, 4, 5],
            "Y_centroid": [5, 4, 3, 2, 1, 0],
        },
        index=[f"cell-{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["A", "B"])
    adata = AnnData(np.arange(12, dtype=float).reshape(6, 2), obs=obs, var=var)
    adata.raw = adata.copy()
    adata.layers["scaled"] = adata.X / 2
    adata.uns["all_markers"] = ["A", "B"]
    return adata


def test_marker_data_supports_raw_layer_and_log(capsys):
    from scimap.plotting.napariGater import get_marker_data

    adata = _adata()
    raw = get_marker_data("A", adata, "raw", log=False, verbose=True)
    scaled = get_marker_data("B", adata, "scaled", log=True, verbose=True)

    np.testing.assert_array_equal(raw["A"], adata.raw.X[:, 0])
    np.testing.assert_allclose(scaled["B"], np.log1p(adata.layers["scaled"][:, 1]))
    assert "Raw data range" in capsys.readouterr().out


def test_auto_contrast_supports_numpy_dask_and_constant_images():
    from scimap.plotting.napariGater import calculate_auto_contrast

    low, high = calculate_auto_contrast(np.arange(100).reshape(10, 10), padding=0)
    assert low < high

    dask_image = da.from_array(np.arange(200).reshape(2, 10, 10), chunks=(1, 5, 5))
    dask_low, dask_high = calculate_auto_contrast(dask_image, padding=0)
    assert dask_low < dask_high

    assert calculate_auto_contrast(np.ones((4, 4)), padding=0) == (1.0, 2.0)


def test_initialize_gates_records_provenance_for_existing_gates():
    from scimap.plotting.napariGater import initialize_gates

    adata = _adata()
    adata.uns["gates"] = pd.DataFrame({"image-1": [1.5, 2.5]}, index=adata.var_names)

    result = initialize_gates(adata, "imageid", "raw", False, False)

    provenance = result.uns["napariGaterProvenance"]
    assert provenance["original_values"]["image-1"] == {"A": 1.5, "B": 2.5}
    assert provenance["manually_adjusted"]["image-1"] == {}


def test_initialize_contrast_settings_initializes_reuses_and_recovers(monkeypatch):
    module = importlib.import_module("scimap.plotting.napariGater")

    adata = _adata()
    monkeypatch.setattr(
        module, "calculate_auto_contrast", lambda image, **kwargs: (2.0, 8.0)
    )

    module.initialize_contrast_settings(adata, np.zeros((2, 3, 3)), ["A", "B"])
    assert adata.uns["image_contrast_settings"]["image-1"]["A"] == {
        "low": 2.0,
        "high": 8.0,
    }

    monkeypatch.setattr(
        module,
        "calculate_auto_contrast",
        Mock(side_effect=AssertionError("should not run")),
    )
    module.initialize_contrast_settings(adata, np.zeros((2, 3, 3)), ["A", "B"])

    monkeypatch.setattr(
        module, "calculate_auto_contrast", Mock(side_effect=ValueError("bad image"))
    )
    module.initialize_contrast_settings(adata, [np.zeros((1, 3, 3))], ["C"])
    assert adata.uns["image_contrast_settings"]["image-1"]["C"] == {
        "low": 0.0,
        "high": 100,
    }


def test_load_image_and_add_channels_helpers(capsys):
    from scimap.plotting.napariGater import (
        add_channels_to_viewer,
        load_image_efficiently,
    )

    assert load_image_efficiently("not-an-image.png") == (None, None, False)

    viewer = Mock()
    add_channels_to_viewer(
        viewer,
        np.zeros((3, 2, 2)),
        ["A", "B", "C"],
        {"A": {"low": 1, "high": 9}, "B": {"low": 2, "high": 8}},
        ["red", "green"],
    )
    kwargs = viewer.add_image.call_args.kwargs
    assert kwargs["colormap"] == ["red", "green", "red"]
    assert kwargs["contrast_limits"][-1] == (0.0, 100)
    assert "Missing contrast settings" in capsys.readouterr().out


def test_voronoi_finite_polygons_reconstructs_regions_and_validates_dimensions():
    from scimap.plotting.voronoi import voronoi_finite_polygons_2d

    points = np.array([[0, 0], [0, 2], [2, 0], [2, 2], [1, 1]], dtype=float)
    regions, vertices = voronoi_finite_polygons_2d(Voronoi(points), radius=10)
    assert len(regions) == len(points)
    assert vertices.shape[1] == 2
    assert all(len(region) >= 3 for region in regions)

    invalid = Mock()
    invalid.points = np.zeros((4, 3))
    with pytest.raises(ValueError, match="Requires 2D input"):
        voronoi_finite_polygons_2d(invalid)


def test_cluster_plots_generates_all_outputs_with_mocked_scanpy(monkeypatch, tmp_path):
    module = importlib.import_module("scimap.plotting.cluster_plots")

    adata = _adata()
    umap_figure = Mock()
    matrix_figure = Mock()
    monkeypatch.setattr(module.sc.pp, "subsample", Mock())
    monkeypatch.setattr(module.sc.pp, "neighbors", Mock())
    monkeypatch.setattr(module.sc.tl, "umap", Mock())
    monkeypatch.setattr(module.sc.tl, "rank_genes_groups", Mock())
    monkeypatch.setattr(module.sc.pl, "umap", Mock(return_value=umap_figure))
    monkeypatch.setattr(module.sc.pl, "matrixplot", Mock(return_value=matrix_figure))
    monkeypatch.setattr(module.sc.pl, "rank_genes_groups", Mock())
    monkeypatch.setattr(module.plt, "savefig", Mock())

    module.cluster_plots(adata, "cluster", subsample=3, output_dir=tmp_path)

    module.sc.pp.subsample.assert_called_once_with(adata, n_obs=3)
    umap_figure.savefig.assert_called_once_with(tmp_path / "_umap.pdf")
    matrix_figure.savefig.assert_called_once_with(tmp_path / "_matrixplot.pdf")
    module.plt.savefig.assert_called_once_with(
        tmp_path / "_ranked_markers_per_cluster.pdf"
    )


def test_cluster_plots_handles_plotting_failures(monkeypatch, capsys):
    module = importlib.import_module("scimap.plotting.cluster_plots")

    adata = _adata()
    monkeypatch.setattr(
        module.sc.pp, "neighbors", Mock(side_effect=RuntimeError("umap"))
    )
    monkeypatch.setattr(
        module.sc.pl, "matrixplot", Mock(side_effect=RuntimeError("matrix"))
    )
    monkeypatch.setattr(
        module.sc.tl, "rank_genes_groups", Mock(side_effect=RuntimeError("rank"))
    )

    module.cluster_plots(adata, "cluster", subsample=None)

    output = capsys.readouterr().out
    assert "UMAP could not be generated" in output
    assert "Heatmap could not be generated" in output
    assert "Finding differential markers" in output


def test_image_viewer_opens_zarr_and_adds_overlay_layers(monkeypatch, tmp_path):
    module = importlib.import_module("scimap.plotting.image_viewer")

    viewer = Mock()
    napari = Mock()
    napari.Viewer.return_value = viewer
    monkeypatch.setattr(module, "napari", napari, raising=False)
    missing_zarr = tmp_path / "image.zarr"

    module.image_viewer(
        str(missing_zarr),
        _adata(),
        overlay="cluster",
        overlay_category=["a", "b"],
        point_color={"a": "red"},
        flip_y=False,
    )

    viewer.open.assert_called_once_with(
        str(missing_zarr),
        multiscale=True,
        visible=False,
        name=["A", "B"],
    )
    assert viewer.add_points.call_count == 2
    assert viewer.add_points.call_args_list[0].kwargs["face_color"] == "red"
    assert viewer.add_points.call_args_list[1].kwargs["face_color"] == "white"


def test_gate_finder_creates_a_layer_for_each_threshold(monkeypatch, tmp_path):
    module = importlib.import_module("scimap.plotting.gate_finder")

    viewer = Mock()
    napari = Mock()
    napari.Viewer.return_value = viewer
    monkeypatch.setattr(module, "napari", napari, raising=False)

    with pytest.warns(FutureWarning, match="deprecated"):
        module.gate_finder(
            str(tmp_path / "image.zarr"),
            _adata(),
            marker_of_interest="A",
            layer=None,
            log=False,
            from_gate=0,
            to_gate=2,
            increment=1,
            markers="B",
            flip_y=False,
        )

    viewer.open.assert_called_once()
    assert viewer.add_points.call_count == 2
    assert [call.kwargs["name"] for call in viewer.add_points.call_args_list] == [
        "gate-0",
        "gate-1",
    ]
