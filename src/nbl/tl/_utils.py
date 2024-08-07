from collections import OrderedDict

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from more_itertools import one
from numpy.typing import NDArray
from skimage.measure import regionprops_table
from spatialdata.models import X, Y


def _rp_stats(label: NDArray, properties: list[str]) -> NDArray:
    """Generates a table of region properties for a label image.

    Parameters
    ----------
    label : NDArray
        The label image.
    properties : list[str]
        The properties to compute.

    Returns
    -------
    NDArray
        The NumPy array of region properties.
    """
    _rp = OrderedDict(regionprops_table(label_image=label, intensity_image=None, properties=properties))
    _rp_da: NDArray = da.from_array(x=np.array(list(_rp.values()))).T

    return _rp_da


def _rp_stats_table_fov(
    sdata: sd.SpatialData,
    label_type: str,
    table_name: str,
    instance_key: str,
    properties: list[str],
    properties_names: list[str],
) -> sd.SpatialData:
    """
    Computes and adds region properties statistics to a SpatailData AnnData table.

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object..
    label_type : str
        The type of label used to identify regions of interest (e.g., 'whole_cell', 'nucleus').
    table_name : str
        The name of the table within the SpatialData object where the region properties will be added.
    instance_key : str
        The key that uniquely identifies instances within the label type.
    properties : list[str]
        A list of region properties to compute (e.g., area, perimeter).
    properties_names : list[str]
        The names of the properties to be used in the output table. These should correspond
        to the computed properties in the same order.

    Returns
    -------
    sd.SpatialData
        The updated SpatialData object with the computed region properties added to the specified table.

    Notes
    -----
    This function computes specified region properties for each region of interest in the spatial data,
    and integrates these properties into the specified table within the SpatialData object.

    - It uses xarray's `apply_ufunc` to parallelize the calculation.
    - The computed properties are merged with the existing table in the SpatialData object.
    """
    n_obs = sdata.tables[table_name].n_obs
    coord = one(sdata.coordinate_systems)

    n_properties = len(properties_names)
    rp = xr.apply_ufunc(
        _rp_stats,
        sdata.labels[f"{coord}_{label_type}"],
        kwargs={"properties": properties},
        input_core_dims=[[Y, X]],
        output_core_dims=[[instance_key, "property"]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {instance_key: n_obs, "property": n_properties}},
    )

    sdata_table: ad.AnnData = sdata.tables[table_name]

    rp_table = pd.DataFrame(
        data=rp.data,
        columns=properties_names,
        index=sdata_table.obs_names,
    )
    if "centroid" in properties:
        rp_table_centroids: pd.DataFrame = rp_table[[Y, X]]
        rp_table: pd.DataFrame = rp_table.drop(columns=[Y, X])
        sdata_table.obsm["spatial"] = rp_table_centroids.to_numpy()
    sdata_table.obs = sdata_table.obs.merge(right=rp_table, right_on="label", left_on=instance_key).drop(
        columns=["label"]
    )
    return sdata


def _set_unique_obs_names(
    sdata: sd.SpatialData,
    coord: str,
    table_name: str,
) -> sd.SpatialData:
    """Sets unique observation names by combining the coordinate system name and observation index.

    This function modifies the observation names in the specified table of the `SpatialData`
    object to be unique by prefixing each observation index with the coordinate system name.

    Parameters
    ----------
    sdata : sd.SpatialData
        The `SpatialData` object containing tables.
    coord : str
        The name of the coordinate system.
    table_name : str
        The name of the table within the `SpatialData` object.

    Returns
    -------
    sd.SpatialData
        The modified `SpatialData` object with unique observation names in the specified table.
    """
    obs_integer_range = range(1, sdata.tables[table_name].n_obs + 1)
    sdata.tables[table_name].obs_names = pd.Index(
        data=[f"{coord}_{obs_index_val}" for obs_index_val in obs_integer_range],
        dtype=str,
    )
    return sdata


def _strip_var_names(sdata: sd.SpatialData, table_name: str, agg_func: str) -> sd.SpatialData:
    """Strips prefixes and suffixes from variable names in the specified table.

    This function modifies the variable names in the specified table of the `SpatialData`
    object by removing a specific prefix and suffix related to the aggregation function.

    Parameters
    ----------
    sdata : sd.SpatialData
        The `SpatialData` object containing tables.
    table_name : str
        The name of the table within the `SpatialData` object.
    agg_func : str
        The aggregation function used, which should be removed from the variable names.

    Returns
    -------
    sd.SpatialData
        The modified `SpatialData` object with stripped variable names in the specified table.
    """
    sdata.tables[table_name].var_names = [
        var.removeprefix("channel_").removesuffix(f"_{agg_func}") for var in sdata.tables[table_name].var_names
    ]
    return sdata
