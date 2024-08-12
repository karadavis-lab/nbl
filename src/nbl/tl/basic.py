from itertools import chain

import anndata as ad
import natsort as ns
import numpy as np
import spatialdata as sd
from dask import delayed

from nbl.tl._utils import _rp_stats_table_fov, _set_unique_obs_names, _strip_var_names
from nbl.util import DaskSetupDelayed
from nbl.util.decorators import catch_warnings


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
    sdata : sd.SpatialData
        A SpatialData object.
    label_type : str
        The type of label used to identify regions of interest (e.g., 'whole_cell', 'nucleus').
    table_name : str, optional
        The name of the table within the SpatialData object where the aggregated data will be stored. Default is None.
    region_key : str, optional
        The key that represents regions in the concatenated data. Default is "region".
    instance_key : str, optional
        The key that uniquely identifies instances within the label type. Default is "instance_id".
    agg_func : str, optional
        The aggregation function to be applied (e.g., 'sum', 'mean'). Default is "sum".
    inplace : bool, optional
        If True, modifies the input SpatialData object in place. Otherwise, returns a new SpatialData object.
        Default is True.
    write : bool, optional
        If True, writes the updated table to disk. Default is True.

    Returns
    -------
    None | sd.SpatialData
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
        t1 = t1.persist()
        t2_t3 = delayed(_strip_var_names)(
            sdata=delayed(_set_unique_obs_names)(sdata=t1, coord=coord, table_name=table_name),
            table_name=table_name,
            agg_func=agg_func,
        )
        _tasks.append(t2_t3)

    dask_runner = DaskSetupDelayed(delayed_objects=_tasks)
    _sdatas = dask_runner.compute()

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


def quantile(adata: ad.AnnData, var: str, q: float, filter_adata: bool = False, **kwargs) -> np.floating | ad.AnnData:
    """Computes the qth quantile of a particular `var` in the `AnnData` object.

    Parameters
    ----------
    adata : ad.AnnData
        The `AnnData` object.
    var : str
        The variable to compute the quantile for.
    q : float
        Percentage or sequence of percentages for the percentiles to compute.
        Must be between 0 and 1 inclusive.
    filter_adata: bool
        Whether or not to filter the `obs_names` by the value `v` associated to the qth quantile.
    kwargs: Mapping[str, Any]
        All other keyword arguments are passed to `np.quantile`

    Returns
    -------
    np.floating
        The value corresponding to the qth quantile.
    """
    v: np.floating = np.quantile(a=adata[:, var].X.toarray().squeeze(), q=q, **kwargs)
    if "quantiles" not in adata.uns.keys():
        adata.uns["quantiles"] = {}
    if var not in adata.uns["quantiles"].keys():
        adata.uns["quantiles"] = {var: {}}
    adata.uns["quantiles"][var].update({"quantile": q, "value": v})
    if filter_adata:
        filtered_adata: ad.AnnData = filter_obs_names_by_quantile(adata=adata, var=var, v=v, copy=False)
        return filtered_adata
    else:
        return v


def filter_obs_names_by_quantile(adata: ad.AnnData, var: str, v: float, copy: bool = False) -> ad.AnnData:
    """Filters the `obs_names` index based on the value (v) corresponding to the qth quantile.

    Given an AnnData object A (n_obs x n_vars) we can select the observations with a value greater than or equal to v.

    Parameters
    ----------
    adata : ad.AnnData
        The `AnnData` object.
    var : str
        The variable to filter on.
    v : float
        The value to filter by.
    copy: bool
        Whether or not to make a full copy of the subseted AnnData object

    Returns
    -------
    ad.AnnData
        A subset of the `AnnData` object.
    """
    filtered_idx = adata[:, var].X.toarray().squeeze() >= v
    if copy:
        return adata[filtered_idx, :].copy()
    else:
        return adata[filtered_idx, :]
