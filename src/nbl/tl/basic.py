from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any, Literal, TypedDict, Unpack

import anndata as ad
import natsort as ns
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from dask import delayed
from leidenalg.VertexPartition import MutableVertexPartition
from numpydantic import NDArray

from nbl.util import DaskSetupDelayed
from nbl.util.decorators import catch_warnings
from nbl.util.utils import _extract_layer_from_sdata

from ._utils import _rp_stats_table_fov, _scores, _set_unique_obs_names, _strip_var_names


@catch_warnings
def aggregate_images_by_labels(
    sdata: sd.SpatialData,
    label_type: str,
    table_name: str = None,
    region_key: str = "region",
    instance_key: str = "instance_id",
    agg_func: str = "sum",
    inplace: bool = True,
    write: bool = True,
) -> None | sd.SpatialData:
    """
    Aggregates image data by labels using a specified aggregation function and updates the spatial data table.

    Parameters
    ----------
    sdata
        A SpatialData object.
    label_type
        The type of label used to identify regions of interest (e.g., 'whole_cell', 'nucleus').
    table_name
        The name of the table within the SpatialData object where the aggregated data will be stored. Default is None.
    region_key
        The key that represents regions in the concatenated data. Default is "region".
    instance_key
        The key that uniquely identifies instances within the label type. Default is "instance_id".
    agg_func
        The aggregation function to be applied (e.g., 'sum', 'mean'). Default is "sum".
    inplace
        If True, modifies the input SpatialData object in place. Otherwise, returns a new SpatialData object.
        Default is True.
    write
        If True, writes the updated table to disk. Default is True.

    Returns
    -------
    The updated SpatialData object if inplace is False, otherwise None.

    Notes
    -----
    This function aggregates image data by the specified labels using the provided aggregation function,
    and integrates the aggregated data into the specified table within the SpatialData object.

    - The function handles multiple coordinate systems present in the SpatialData object.
    - The computed aggregated data is merged with the existing table in the SpatialData object.
    - The function uses delayed execution via Dask for parallel processing and improved performance.
    """
    all_coords = ns.natsorted(sdata.coordinate_systems)
    _tasks = []
    for coord in all_coords:
        sdata_coord = delayed(sdata.filter_by_coordinate_system)(coordinate_system=coord)
        t1 = delayed(sdata_coord.aggregate)(
            values=coord,
            by=f"{coord}_{label_type}",
            table_name=table_name,
            target_coordinate_system=coord,
            agg_func=agg_func,
            region_key=region_key,
            instance_key=instance_key,
        )
        t2 = delayed(_set_unique_obs_names)(sdata=t1, coord=coord, table_name=table_name)
        t3 = delayed(_strip_var_names)(sdata=t2, table_name=table_name, agg_func=agg_func)
        _tasks.append(t3)

    # Initialize Dask Runner and compute tasks
    dask_runner = DaskSetupDelayed(delayed_objects=_tasks)
    _sdatas = dask_runner.compute()

    # Concatenate the results and update the sdata table
    adata = sd.concatenate(
        sdatas=_sdatas, region_key=region_key, instance_key=instance_key, concatenate_tables=True
    ).tables[table_name]

    sdata.tables[table_name] = adata

    if write:
        from nbl.util import write_elements

        write_elements(sdata=sdata, elements={"tables": table_name})

    return None if inplace else sdata


def regionprops(
    sdata: sd.SpatialData,
    label_type: str,
    table_name: str,
    region_key: str = "region",
    instance_key: str = "instance_id",
    properties: list[str] | None = None,
    inplace: bool = True,
    write: bool = True,
) -> sd.SpatialData | None:
    """
    Computes region properties for spatial data and updates the specified table with these properties.

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object.
    label_type : str
        The type of label used to identify regions of interest (e.g., 'whole_cell', 'nucleus').
    table_name : str
        The name of the table within the SpatialData object where the statistics will be added.
    region_key : str, optional
        The key that represents regions in the concatenated data, by default "region".
    instance_key : str, optional
        The key that uniquely identifies instances within the label type, by default "instance_id".
    properties : list[str] | None, optional
        A list of region properties to compute (e.g., area, perimeter). If None, defaults to ["label", "centroid"].
    inplace : bool, optional
        If True, modifies the input SpatialData object in place. Otherwise, returns a new SpatialData object.
        Default is True.
    write : bool, optional
        If True, writes the updated table to disk. Default is True.

    Returns
    -------
    sd.SpatialData | None
        The updated SpatialData object if inplace is False, otherwise None.

    Notes
    -----
    This function computes specified region properties for each region of interest in the spatial data,
    and integrates these properties into the specified table within the SpatialData object.

    - The function handles multiple coordinate systems present in the spatial data.
    - The computed properties are merged with the existing table in the SpatialData object.
    - The function uses delayed execution via Dask for parallel processing and improved performance.
    """
    if properties is None:
        properties = ["label", "centroid"]
    all_coords = ns.natsorted(sdata.coordinate_systems)

    if "label" not in properties:
        properties.append("label")

    n_properties = len(properties)

    if "centroid" in properties:
        n_properties += 1

    properties_names = list(chain.from_iterable(("y", "x") if x == "centroid" else (x,) for x in properties.copy()))

    _tasks = []

    for coord in all_coords:
        fov_sdata: sd.SpatialData = delayed(sdata.filter_by_coordinate_system)(
            coordinate_system=coord, filter_tables=True, include_orphan_tables=False
        )
        t1 = delayed(_rp_stats_table_fov)(
            sdata=fov_sdata,
            label_type=label_type,
            table_name=table_name,
            instance_key=instance_key,
            properties=properties,
            properties_names=properties_names,
        )
        t2 = delayed(_set_unique_obs_names)(sdata=t1, coord=coord, table_name=table_name)

        _tasks.append(t2)
    dask_runner = DaskSetupDelayed(delayed_objects=_tasks)

    _sdatas = dask_runner.compute()
    adata: ad.AnnData = sd.concatenate(
        sdatas=_sdatas, region_key=region_key, instance_key=instance_key, concatenate_tables=True
    ).tables[table_name]
    sdata.tables[table_name] = adata

    if write:
        from nbl.util import write_elements

        write_elements(sdata=sdata, elements={"tables": table_name})

    return None if inplace else sdata


def quantile(
    sdata: sd.SpatialData,
    table_name: str,
    var: str,
    q: float | list[float],
    layer: str | None = None,
    inplace: bool = True,
    write: bool = True,
) -> sd.SpatialData | None:
    """Computes the qth quantile of a particular `var` in the `AnnData` object.

    Parameters
    ----------
    sdata
        The SpatialData object.
    table_name
        The name of the table within the SpatialData object.
    layer
        The name of the layer within the AnnData table object.
    var
        The variable / marker to compute the quantile for.
    q
        The quantile to compute.
    inplace, optional
        If True modifies the input SpatialData object in place. Otherwise, returns a new SpatialData object.
        Default is True.
    write, optional
        If True, writes the updated table to disk. Default is True.

    Returns
    -------
    The updated SpatialData object if inplace is False, otherwise None.
    """
    table: ad.AnnData = sdata.tables[table_name]

    if layer:
        v: np.floating = np.quantile(a=table[:, var].layers[layer].toarray().squeeze(), q=q)
    else:
        v: np.floating = np.quantile(a=table[:, var].X.toarray().squeeze(), q=q)
    _add_quantile_to_uns(adata=table, var=var, q=q, v=v)

    if write:
        from nbl.util import write_elements

        write_elements(sdata=sdata, elements={"tables": table_name})

    return None if inplace else sdata


def _add_quantile_to_uns(adata: ad.AnnData, var: str, q: float | list[float], v: float | list[float]) -> None:
    if "quantiles" not in adata.uns.keys():
        adata.uns["quantiles"] = {}
    if var not in adata.uns["quantiles"].keys():
        adata.uns["quantiles"] = {var: {}}
    for q_val, v_val in zip(q, v, strict=False):
        adata.uns["quantiles"][var].update({f"{q_val}": v_val})


def filter_obs_names_by_quantile(
    sdata: sd.SpatialData,
    table_name: str,
    var: str,
    q: float,
    layer: str | None = None,
    method: Literal["lower", "upper"] = "lower",
) -> ad.AnnData:
    """Filters the `obs_names` index based on the value (v) corresponding to the qth quantile.

    Parameters
    ----------
    sdata
        The SpatialData object.
    table_name
        The name of the table within the SpatialData object.
    var
        The variable to filter on.
    q
        The quantile to filter on.
    layer
        The name of the layer within the AnnData table object.
    method
        The method to use for filtering. Either "lower" or "upper". Default is "lower".

    Returns
    -------
    A filtered AnnData object.
    """
    table: ad.AnnData = sdata.tables[table_name]
    v: float = table.uns["quantiles"][var][f"{q}"]
    if layer:
        X: NDArray = table[:, var].layers[layer].toarray().squeeze()
    else:
        X: NDArray = table[:, var].X.toarray().squeeze()

    if method == "lower":
        filtered_obs_names: pd.Index = table.obs_names[X < v]
    elif method == "upper":
        filtered_obs_names: pd.Index = table.obs_names[X >= v]

    subset_table: ad.AnnData = table[filtered_obs_names, :]
    sdata.update_annotated_regions_metadata(table=subset_table)
    return subset_table


def compute_marker_means(
    element: sd.SpatialData | ad.AnnData,
    table_name: str,
    marker_groups: Mapping[str, str | list[str]],
    layer: str | None = None,
    inplace: bool = False,
) -> sd.SpatialData | ad.AnnData:
    """Computes the mean expression for each marker group.

    Parameters
    ----------
    element
        The SpatialData or AnnData object.
    table_name
        The name of the table within the SpatialData object.
    marker_groups
        A mapping of marker groups to marker names.
    layer
        The name of the layer within the AnnData table object.
    inplace
        If True modifies the input SpatialData object in place. Otherwise, returns a new SpatialData object.
        Default is False.

    Returns
    -------
    The updated SpatialData object if inplace is False, otherwise None.
    """
    if isinstance(element, sd.SpatialData):
        table: ad.AnnData = element.tables[table_name]
    else:
        table: ad.AnnData = element

    for marker_group, markers in marker_groups.items():
        if layer:
            X: NDArray = table[:, markers].layers[layer].toarray().squeeze()
        else:
            X: NDArray = table[:, markers].X.toarray().squeeze()
        mean_score: NDArray = np.mean(X, axis=1)
        table.obs[f"{marker_group}_mean"] = mean_score

    return None if inplace else element


def compute_score(
    sdata: sd.SpatialData,
    table_name: str,
    obs_1: str,
    obs_2: str,
    score_method: str,
    score_col_name: str | None = None,
    inplace: bool = False,
    eps: float = 1e-10,
) -> sd.SpatialData | None:
    """Computes a score from two observation columns in a given table of the SpatialData object.

    Parameters
    ----------
    sdata
        The SpatialData object.
    table_name
        The name of the table within the SpatialData object.
    obs_1
        The name of the first observation column.
    obs_2
        The name of the second observation column.
    score_method
        The name of the score method to use. Options are "ratio", "normalized_difference", "log_ratio", "scaled_difference".
    score_col_name
        The name of the score column to use. If None, defaults to "{score_method}_{obs_1}_{obs_2}_score".
    inplace
        If True modifies the input SpatialData object in place. Otherwise, returns a new SpatialData object.
        Default is False.
    eps
        The epsilon value to use for numerical stability, by default 1e-10.

    Returns
    -------
    The updated SpatialData object if inplace is False, otherwise None.
    """
    table: ad.AnnData = sdata.tables[table_name]
    _score_func = _scores[score_method]
    if score_col_name is None:
        score_col_name = f"{score_method}_{obs_1}_{obs_2}_score"
    table.obs[score_col_name] = _score_func(x1=table.obs[obs_1], x2=table.obs[obs_2], eps=eps)
    return None if inplace else sdata


class DiffmapKwargs(TypedDict):
    """Keyword arguments for the Diffmap function from Scanpy."""

    n_comps: int = 2
    random_state: int | None | Any = 0


def diffmap(
    sdata: sd.SpatialData,
    table_name: str,
    layer: str | None = None,
    neighbors_key: str | None = None,
    vars: Sequence[str] | None = None,
    inplace: bool = True,
    write: bool = True,
    **sc_diffmap_kwargs: Unpack[DiffmapKwargs],
) -> None | sd.SpatialData:
    """Computes the diffusion maps for a given table in the SpatialData object.

    Parameters
    ----------
    sdata
        _description_
    table_name
        _description_
    layer, optional
        _description_, by default None
    neighbors_key, optional
        _description_, by default None
    inplace, optional
        _description_, by default True
    write, optional
        _description_, by default True

    Returns
    -------
        _description_
    """
    table: ad.AnnData = _extract_layer_from_sdata(sdata=sdata, vars=vars, table_name=table_name, layer=layer)

    q: ad.AnnData = sc.tl.diffmap(adata=table, neighbors_key=neighbors_key, copy=True, **sc_diffmap_kwargs)

    sdata.tables[table_name].obsm[f"{layer}_diffmap"] = q.obsm["X_diffmap"]
    sdata.tables[table_name].uns[f"{layer}_diffmap_evals"] = q.uns["diffmap_evals"]

    return None if inplace else sdata


class LeidenKwargs(TypedDict):
    """Keyword arguments for the Leiden function from Scanpy."""

    resolution: float = 1
    restrict_to: tuple[str, Sequence[str]] | None = None
    random_state: int | None | Any = 0
    adjacency: Any | None = None
    directed: bool | None = None
    use_weights: bool = True
    n_iterations: int = -1
    partition_type: type[MutableVertexPartition] | None = None
    flavor: Literal["leidenalg", "igraph"] = "leidenalg"


def leiden(
    sdata: sd.SpatialData,
    table_name: str,
    layer: str,
    neighbors_key: str,
    key_added: str,
    vars: Sequence[str] | None = None,
    inplace: bool = True,
    write: bool = True,
    **sc_leiden_kwargs: Unpack[LeidenKwargs],
) -> None | sd.SpatialData:
    """Compute Leiden clustering algorithm for a given table in the SpatialData object.

    Parameters
    ----------
    sdata
        _description_
    table_name
        _description_
    layer
        _description_
    neighbors_key
        _description_
    key_added
        _description_
    vars, optional
        _description_, by default None
    inplace, optional
        _description_, by default True
    write, optional
        _description_, by default True

    Returns
    -------
        _description_
    """
    table: ad.AnnData = _extract_layer_from_sdata(sdata=sdata, vars=vars, table_name=table_name, layer=layer)
    q: ad.AnnData = sc.tl.leiden(
        adata=table, key_added=key_added, neighbors_key=neighbors_key, copy=True, **sc_leiden_kwargs
    )

    sdata.tables[table_name].obs = sdata.tables[table_name].obs.merge(
        right=q.obs[key_added], left_index=True, right_index=True
    )
    sdata.tables[table_name].uns[key_added] = q.uns[key_added]

    return None if inplace else sdata


class UmapKwargs(TypedDict):
    """Keyword arguments for the Umap function from Scanpy."""

    min_dist: float = 0.5
    spread: float = 1
    n_components: int = 2
    maxiter: int | None = None
    alpha: float = 1
    gamma: float = 1
    negative_sample_rate: int = 5
    init_pos: NDArray | None = "spectral"
    random_state: Any = 0
    a: float | None = None
    b: float | None = None


def umap(
    sdata: sd.SpatialData,
    table_name: str,
    layer: str,
    neighbors_key: str,
    vars: Sequence[str] | None = None,
    inplace: bool = True,
    write: bool = True,
    **sc_umap_kwargs: Unpack[UmapKwargs],
) -> ad.AnnData | None:
    """Compute UMAP embeddings for a given table in the SpatialData object.

    Parameters
    ----------
    sdata
        _description_
    table_name
        _description_
    layer
        _description_
    neighbors_key
        _description_
    vars, optional
        _description_, by default None
    inplace, optional
        _description_, by default True
    write, optional
        _description_, by default True

    Returns
    -------
        _description_
    """
    table = _extract_layer_from_sdata(sdata=sdata, vars=vars, table_name=table_name, layer=layer)
    q = sc.tl.umap(adata=table, neighbors_key=neighbors_key, copy=True, **sc_umap_kwargs)
    return q
