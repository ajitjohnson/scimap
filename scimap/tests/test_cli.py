from pathlib import Path
from unittest.mock import Mock

import pytest


def test_clustering_pass_only_imports_data(monkeypatch, tmp_path):
    import scimap.cli._scimap_mcmicro as module

    importer = Mock()
    monkeypatch.setattr(module.pp, "mcmicro_to_scimap", importer)

    result = module.clustering(
        [
            "program",
            "cells.csv",
            "--output",
            str(tmp_path),
            "--clustering-method",
            "pass",
        ]
    )

    assert result is None
    importer.assert_called_once_with(
        feature_table_path="cells.csv", output_dir=str(tmp_path)
    )


@pytest.mark.parametrize("method", ["kmeans", "spatial"])
def test_clustering_dispatches_expression_and_spatial_workflows(
    monkeypatch, tmp_path, method
):
    import scimap.cli._scimap_mcmicro as module

    def create_adata(*, feature_table_path, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "cells.h5ad").touch()

    monkeypatch.setattr(module.pp, "mcmicro_to_scimap", create_adata)
    monkeypatch.setattr(module.tl, "cluster", Mock())
    monkeypatch.setattr(module.tl, "spatial_expression", Mock())
    monkeypatch.setattr(module.tl, "spatial_cluster", Mock())
    monkeypatch.setattr(module.pl, "cluster_plots", Mock())

    result = module.clustering(
        [
            "program",
            "cells.csv",
            "--output",
            str(tmp_path),
            "--clustering-method",
            method,
        ]
    )

    assert result == 0
    output = tmp_path / method
    if method == "spatial":
        module.tl.spatial_expression.assert_called_once()
        module.tl.spatial_cluster.assert_called_once()
        module.tl.cluster.assert_not_called()
    else:
        module.tl.cluster.assert_called_once()
        module.pl.cluster_plots.assert_called_once()


def test_clustering_rejects_non_csv_input():
    from scimap.cli._scimap_mcmicro import clustering

    with pytest.raises(AssertionError, match="must be a csv"):
        clustering(["program", "cells.tsv"])


def test_merge_reports_empty_directory(tmp_path):
    from scimap.cli._scimap_mcmicro import merge

    assert merge(["program", str(tmp_path)]) == 1


def test_merge_combines_converts_and_deletes_inputs(monkeypatch, tmp_path):
    import scimap.cli._scimap_mcmicro as module

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    first = input_dir / "first.h5ad"
    second = input_dir / "second.h5ad"
    first.touch()
    second.touch()

    def merge_files(*, adata, output_dir):
        output = Path(output_dir) / "combined_adata.h5ad"
        output.touch()

    def export_csv(*, adata, output_dir):
        (Path(output_dir) / "combined_adata.csv").touch()

    monkeypatch.setattr(module.hl, "merge_adata_obs", merge_files)
    monkeypatch.setattr(module.hl, "scimap_to_csv", export_csv)

    result = module.merge(
        [
            "program",
            str(input_dir),
            "--output",
            str(output_dir),
            "--csv",
            "--delete-merged",
        ]
    )

    assert result == 0
    assert (output_dir / "combined_adata.h5ad").exists()
    assert (output_dir / "combined_adata.csv").exists()
    assert not first.exists()
    assert not second.exists()


def test_mcmicro_wrap_single_method_and_csv(monkeypatch, tmp_path):
    import scimap.cli._scimap_mcmicro as module

    method_dir = tmp_path / "kmeans"
    method_dir.mkdir()
    h5ad = method_dir / "cells.h5ad"
    h5ad.touch()
    clustering = Mock(return_value=0)
    export = Mock()
    monkeypatch.setattr(module, "clustering", clustering)
    monkeypatch.setattr(module.hl, "scimap_to_csv", export)

    result = module.mcmicro_wrap(
        [
            "program",
            "cells.csv",
            "--output",
            str(tmp_path),
            "--method",
            "kmeans",
            "--csv",
        ]
    )

    assert result == 0
    clustering.assert_called_once()
    export.assert_called_once_with(adata=str(h5ad), output_dir=method_dir)


def test_mcmicro_wrap_all_methods_merges_and_moves_plots(monkeypatch, tmp_path):
    import scimap.cli._scimap_mcmicro as module

    plot = tmp_path / "kmeans" / "plot.pdf"
    plot.parent.mkdir()
    plot.touch()
    monkeypatch.setattr(module, "clustering", Mock(return_value=0))
    monkeypatch.setattr(module, "merge", Mock(return_value=0))
    monkeypatch.setattr(module.time, "sleep", Mock())

    assert module.mcmicro_wrap(["program", "cells.csv", "--output", str(tmp_path)]) == 0
    assert module.clustering.call_count == 4
    module.merge.assert_called_once()
    assert (tmp_path / "plots" / "kmeans" / "plot.pdf").exists()


def test_mcmicro_wrap_propagates_merge_failure(monkeypatch, tmp_path):
    import scimap.cli._scimap_mcmicro as module

    monkeypatch.setattr(module, "clustering", Mock(return_value=0))
    monkeypatch.setattr(module, "merge", Mock(return_value=7))

    assert module.mcmicro_wrap(["program", "cells.csv", "--output", str(tmp_path)]) == 7
