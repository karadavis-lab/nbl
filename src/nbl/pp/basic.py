from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict, Unpack

import anndata as ad
import numpy as np
import scanpy as sc
import spatialdata as sd
from numpydantic import NDArray
from scipy import sparse

from nbl.util import _extract_layer_from_sdata, write_elements


def arcsinh_transform(
    sdata: sd.SpatialData,
    table_names: str | list[str] = None,
    shift_factor: int | float | Mapping[str, int | float] = 0,
    scale_factor: int | float | Mapping[str, int | float] = 150,
    method: Literal["replace", "layer", "new table"] = "new table",
    write: bool = True,
    inplace: bool = True,
) -> None | sd.SpatialData:
    r"""Apply the arcsinh transformation to the data in the specified tables of the SpatialData object.

    Parameters
    ----------
    sdata
        The SpatialData object containing the data.
    table_names
        The name(s) of the table(s) to transform. If None, all tables will be transformed, by default None.
    shift_factor
        The shift factor to be added to the data before applying the transformation, by default 0.
    scale_factor
        The scale factor to be multiplied with the data after applying the transformation, by default 150.
    method
        Whether to replace the original in `AnnData.X` with the transformed data, to add it as a new layer, or to create a new table, by default "new table".
    write
        Whether to write the transformed tables back to the SpatialData object's Zarr store, by default True.
    inplace
        Whether to modify the SpatialData object in-place or return a new object, by default True.

    Returns
    -------
    If inplace is `True`, returns `None`. If inplace is `False`, returns the modified `SpatialData` object.


    Notes
    -----
    The `shift_factor` and `scale_factor` parameters are used to apply the arcsinh transformation to the data. The transformation is applied as follows:

    .. math::
        \begin{equation}
            \arcsinh{(shift_factor + (\mathbf{X} \cdot scale_factor)}
        \end{equation}


    Examples
    --------
    .. code-block:: python
        sdata = sd.SpatialData()

    """
    table_keys = [table_names] if isinstance(table_names, str) else table_names

    for table in table_keys:
        adata: ad.AnnData = sdata.tables[table]
        if sparse.issparse(adata.X):
            transformed_X: NDArray = np.arcsinh(shift_factor + (adata.X.toarray() * scale_factor))
        else:
            transformed_X: NDArray = np.arcsinh(shift_factor + (adata.X * scale_factor))

        if method == "replace":
            adata.X = transformed_X
        elif method == "layer":
            adata.layers[f"arcsinh_shift_{shift_factor}_scale_{scale_factor}"] = transformed_X
        elif method == "new table":
            new_table: ad.AnnData = adata.copy()
            new_table.X = transformed_X
            sdata.tables[f"arcsinh_shift_{shift_factor}_scale_{scale_factor}"] = new_table

    if write:
        if method in ["replace", "layer"]:
            write_elements(sdata=sdata, elements={"tables": table_keys})
        elif method == "new table":
            write_elements(sdata=sdata, elements={"tables": [f"arcsinh_shift_{shift_factor}_scale_{scale_factor}"]})
    return None if inplace else sdata


def normalize_by_area(
    sdata: sd.SpatialData,
    table_names: str | list[str] = None,
    method: Literal["replace", "layer", "new table"] = "new table",
    inplace: bool = True,
    write: bool = True,
):
    """Normalize the data by the area of each cell.

    Parameters
    ----------
    sdata
        The SpatialData object.
    table_names
        The name(s) of the table(s) to transform. If None, all tables will be transformed, by default None.
    inplace
        Whether to modify the SpatialData object in-place or return a new object, by default True.
    write
        Whether to write the transformed tables back to the SpatialData object's Zarr store, by default True.

    Returns
    -------
    The updated SpatialData object if inplace is False, otherwise None.
    """
    table_keys = [table_names] if isinstance(table_names, str) else table_names

    for table in table_keys:
        adata: ad.AnnData = sdata.tables[table]
        area_normalized_X = sparse.csr_matrix(adata.X / adata.obs["area"].to_numpy().reshape(-1, 1))

        if method == "replace":
            adata.X = area_normalized_X
        elif method == "layer":
            adata.layers["area_normalized"] = area_normalized_X
        elif method == "new table":
            new_table = adata.copy()
            new_table.X = area_normalized_X
            sdata.tables["area_normalized"] = new_table
    return None if inplace else sdata


class NeighborsKwargs(TypedDict):
    """Keyword arguments for the Neighbors function from Scanpy."""

    n_neighbors: int
    n_pcs: int
    use_rep: str
    knn: bool = True
    method: Literal["umap", "gauss"] = "umap"
    transformer: str | None | Any = None
    metric: Literal[
        "cityblock",
        "cosine",
        "euclidean",
        "l1",
        "l2",
        "manhattan",
        "braycurtis",
        "canberra",
        "chebyshev",
        "correlation",
        "dice",
        "hamming",
        "jaccard",
        "kulsinski",
        "mahalanobis",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "sqeuclidean",
        "yule",
    ] = "euclidean"
    metric_kwds: Mapping
    random_state: int | None | Any = 0


def neighbors(
    sdata: sd.SpatialData,
    table_name: str,
    layer: str | None = None,
    vars: Sequence[str] | None = None,
    key_added: str | None = None,
    inplace: bool = True,
    write: bool = True,
    **sc_neighbors_kwargs: Unpack[NeighborsKwargs],
) -> sd.SpatialData:
    """Computes the neighbors of a table in a SpatialData object.

    Parameters
    ----------
    sdata
        _description_
    table_name
        _description_
    layer, optional
        _description_, by default None
    vars, optional
        _description_, by default None
    key_added, optional
        _description_, by default None
    inplace, optional
        _description_, by default True
    write, optional
        _description_, by default True

    Returns
    -------
        _description_
    """
    q: ad.AnnData = sc.pp.neighbors(
        adata=_extract_layer_from_sdata(sdata=sdata, vars=vars, table_name=table_name, layer=layer),
        key_added=key_added,
        copy=True,
        **sc_neighbors_kwargs,
    )

    sdata.tables[table_name].obsp[f"{key_added}_connectivities"] = q.obsp[f"{key_added}_connectivities"]
    sdata.tables[table_name].obsp[f"{key_added}_distances"] = q.obsp[f"{key_added}_distances"]
    sdata.tables[table_name].uns[key_added] = q.uns[key_added]

    if write:
        write_elements(sdata=sdata, elements={"tables": table_name})
    return None if inplace else sdata
