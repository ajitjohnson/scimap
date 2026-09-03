import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture
def rescale_adata():
    values = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [4.0, 1.0],
            [5.0, 3.0],
            [6.0, 5.0],
        ]
    )
    adata = AnnData(
        values,
        obs=pd.DataFrame(
            {"imageid": ["one"] * 3 + ["two"] * 3},
            index=[f"cell-{index}" for index in range(len(values))],
        ),
        var=pd.DataFrame(index=["A", "B"]),
    )
    adata.raw = adata.copy()
    return adata


def test_rescale_uses_image_specific_manual_gates(rescale_adata):
    from scimap.preprocessing.rescale import rescale

    gates = pd.DataFrame(
        {"one": [1.0, 2.0], "two": [5.0, 3.0]},
        index=pd.Index(["A", "B"], name="markers"),
    )
    result = rescale(rescale_adata, gate=gates, log=False, verbose=False)

    assert result is rescale_adata
    assert result.X.shape == (6, 2)
    assert np.isfinite(result.X).all()
    assert ((result.X >= 0) & (result.X <= 1)).all()
    assert result.uns["gates"].columns.tolist() == ["one", "two"]
    assert result.uns["gates"].loc["A", "one"] == 1.0


def test_rescale_supports_global_gates_and_initializes_raw_data(rescale_adata):
    from scimap.preprocessing.rescale import rescale

    rescale_adata.raw = None
    gates = pd.DataFrame({"gates": [1.0, 2.0]}, index=["A", "B"])

    result = rescale(rescale_adata, gate=gates, log=True, verbose=False)

    assert result.raw is not None
    assert result.uns["gates"].loc["A"].eq(1.0).all()
    assert result.uns["gates"].loc["B"].eq(2.0).all()
    assert np.isfinite(result.X).all()


def test_rescale_overrides_manual_gates_for_failed_markers(rescale_adata):
    from scimap.preprocessing.rescale import rescale

    gates = pd.DataFrame(
        {"one": [0.5, 0.5], "two": [0.5, 0.5]},
        index=pd.Index(["A", "B"], name="markers"),
    )
    result = rescale(
        rescale_adata,
        gate=gates,
        failed_markers={"all": "B", "one": ["A"]},
        log=False,
        verbose=False,
    )

    assert result.uns["gates"].loc["A", "one"] == 2.0
    assert result.uns["gates"].loc["B", "one"] == 4.0
    assert result.uns["gates"].loc["B", "two"] == 5.0


@pytest.mark.parametrize("method", ["all", "by_image"])
def test_rescale_calculates_missing_gates_with_gmm(rescale_adata, method):
    from scimap.preprocessing.rescale import rescale

    result = rescale(
        rescale_adata,
        gate=None,
        method=method,
        log=False,
        random_state=2,
        gmm_components=2,
        verbose=False,
    )

    assert result.uns["gates"].notna().all().all()
    assert np.isfinite(result.X).all()
    assert ((result.X >= 0) & (result.X <= 1)).all()


def test_rescale_rejects_non_mapping_failed_markers(rescale_adata):
    from scimap.preprocessing.rescale import rescale

    gates = pd.DataFrame({"gates": [1.0, 2.0]}, index=["A", "B"])
    with pytest.raises(ValueError, match="python dictionary"):
        rescale(
            rescale_adata,
            gate=gates,
            failed_markers=["A"],
            log=False,
            verbose=False,
        )
