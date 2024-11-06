from collections.abc import Mapping
from itertools import chain
from typing import Literal

import anndata as ad
import natsort as ns
import numpy as np
import pandas as pd
import spatialdata as sd
from dask import delayed
from more_itertools import one
from numpydantic import NDArray

from nbl.util import write_elements
from nbl.util.decorators import catch_warnings

from ._utils import ScoreFunctions, _rp_stats_table_fov, _set_unique_obs_names, _strip_var_names


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

    filtered_sdatas = {coord: sdata.filter_by_coordinate_system(coordinate_system=coord) for coord in all_coords}

    @delayed
    @catch_warnings
    def process_fov(fov_sdata: sd.SpatialData, fov: str):
        result = fov_sdata.aggregate(
            values=fov,
            by=f"{fov}_{label_type}",
            table_name=table_name,
            target_coordinate_system=fov,
            agg_func=agg_func,
            region_key=region_key,
            instance_key=instance_key,
        )
        result = _set_unique_obs_names(result, fov, table_name)
        result = _strip_var_names(result, table_name, agg_func)
        return result

    tasks = [process_fov(filtered_sdatas[coord], coord) for coord in all_coords]

    optimized_tasks = delayed(lambda x: x)(tasks)

    _sdatas = optimized_tasks.compute()
    adata = sd.concatenate(
        sdatas=_sdatas, region_key=region_key, instance_key=instance_key, concatenate_tables=True
    ).tables[table_name]

    sdata.tables[table_name] = adata

    if write:
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
    sdata
        A SpatialData object.
    label_type
        The type of label used to identify regions of interest (e.g., 'whole_cell', 'nucleus').
    table_name
        The name of the table within the SpatialData object where the statistics will be added.
    region_key
        The key that represents regions in the concatenated data, by default "region".
    instance_key
        The key that uniquely identifies instances within the label type, by default "instance_id".
    properties
        A list of region properties to compute (e.g., area, perimeter). If None, defaults to ["label", "centroid"].
    inplace
        If `True`, modifies the input SpatialData object in place. Otherwise, returns a new SpatialData object.
        Default is `True`.
    write
        If `True`, writes the updated table to disk. Default is `True`.

    Returns
    -------
    The updated `SpatialData` object if inplace is `False`, otherwise `None`.

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

    filtered_sdatas = {coord: sdata.filter_by_coordinate_system(coordinate_system=coord) for coord in all_coords}

    tasks = [
        delayed(_regionprops_fov)(
            filtered_sdatas[coord], label_type, table_name, instance_key, properties, properties_names
        )
        for coord in all_coords
    ]

    optimized_tasks = delayed(lambda x: x)(tasks)

    _sdatas = optimized_tasks.compute()

    adata: ad.AnnData = sd.concatenate(
        sdatas=_sdatas, region_key=region_key, instance_key=instance_key, concatenate_tables=True
    ).tables[table_name]

    sdata.tables[table_name] = adata

    if write:
        write_elements(sdata=sdata, elements={"tables": table_name})

    return None if inplace else sdata


def _regionprops_fov(
    sdata: sd.SpatialData,
    label_type: str,
    table_name: str,
    instance_key: str,
    properties: list[str] | None = None,
    properties_names: list[str] | None = None,
) -> sd.SpatialData:
    _sdata = _rp_stats_table_fov(
        sdata=sdata,
        label_type=label_type,
        table_name=table_name,
        instance_key=instance_key,
        properties=properties,
        properties_names=properties_names,
    )
    _set_unique_obs_names(sdata=_sdata, coord=one(sdata.coordinate_systems), table_name=table_name)
    return _sdata


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
    inplace
        If True modifies the input SpatialData object in place. Otherwise, returns a new SpatialData object.
        Default is True.
    write
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
    _score_func = ScoreFunctions()[score_method]
    if score_col_name is None:
        score_col_name = f"{score_method}_{obs_1}_{obs_2}_score"
    table.obs[score_col_name] = _score_func(x1=table.obs[obs_1], x2=table.obs[obs_2], eps=eps)
    return None if inplace else sdata
