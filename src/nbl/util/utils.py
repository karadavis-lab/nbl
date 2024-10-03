from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata as sd
from matplotlib import ticker
from more_itertools import first


def _replace_element(sdata: sd.SpatialData, element_name: str):
    """_summary_.

    Parameters
    ----------
    sdata : sd.SpatialData
        _description_
    element_name : str
        _description_
    """
    _temp_name = f"{element_name}__temp"
    sdata[_temp_name] = sdata[element_name]
    sdata.write_element(element_name=_temp_name)

    # Delete the original element, and rewrite it (element must be in memory)
    sdata.delete_element_from_disk(element_name=element_name)
    sdata.write_element(element_name=element_name)

    # Remove the copy
    sdata.delete_element_from_disk(_temp_name)
    _element_type = sdata._element_type_from_element_name(_temp_name)
    del getattr(sdata, _element_type)[_temp_name]


def _write_element(
    sdata: sd.SpatialData,
    element_name: str,
    element_type: str,
    not_in_zarr_store: list[str],
    in_zarr_store: list[str],
):
    """_summary_.

    Parameters
    ----------
    sdata : sd.SpatialData
        _description_
    element_name : str
        _description_
    element_type : str
        _description_
    not_in_zarr_store : list[str]
        _description_
    in_zarr_store : list[str]
        _description_
    """
    if element_type == "tables":
        _update_and_sync_table(sdata=sdata, table_name=element_name)

    if f"{element_type}/{element_name}" in not_in_zarr_store and not in_zarr_store:
        sdata.write_element(element_name=element_name, overwrite=False)
    else:
        _replace_element(sdata=sdata, element_name=element_name)


def _update_and_sync_table(sdata: sd.SpatialData, table_name: str):
    sdata.tables[table_name] = sdata.update_annotated_regions_metadata(table=sdata.tables[table_name])


def write_elements(
    sdata: sd.SpatialData,
    elements: Mapping[str, str | list[str]],
) -> None:
    """_summary_.

    Parameters
    ----------
    sdata : sd.SpatialData
        _description_
    elements : Mapping[str, str  |  list[str]]
        _description_
    """
    not_in_zarr_store, _in_zarr_store = sdata._symmetric_difference_with_zarr_store()
    for e_type, e_name in elements.items():
        match e_name:
            case str():
                _write_element(
                    sdata=sdata,
                    element_name=e_name,
                    element_type=e_type,
                    not_in_zarr_store=not_in_zarr_store,
                    in_zarr_store=_in_zarr_store,
                )
            case list():
                for e in e_name:
                    _write_element(
                        sdata=sdata,
                        element_name=e,
                        element_type=e_type,
                        not_in_zarr_store=not_in_zarr_store,
                        in_zarr_store=_in_zarr_store,
                    )
    sdata.write_consolidated_metadata()


def _remove_x_axis_ticks(ax: plt.Axes) -> None:
    """_summary_.

    Parameters
    ----------
    ax : plt.Axes
        _description_
    """
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def _remove_y_axis_ticks(ax: plt.Axes) -> None:
    """_summary_.

    Parameters
    ----------
    ax : plt.Axes
        _description_
    """
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def _set_locator_formatter(ax: plt.Axes, axis: Literal["x", "y", "xy", "yx"]) -> None:
    """_summary_.

    Parameters
    ----------
    ax : plt.Axes
        _description_
    axis : Literal[&quot;x&quot;, &quot;y&quot;, &quot;xy&quot;, &quot;yx&quot;]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    match axis:
        case "x":
            _remove_x_axis_ticks(ax)
        case "y":
            _remove_y_axis_ticks(ax)
        case "xy" | "yx":
            _remove_x_axis_ticks(ax)
            _remove_y_axis_ticks(ax)
        case _:
            raise ValueError("axis must be 'x', 'y' or 'xy' or 'yx'")


def remove_ticks(f: plt.Figure | plt.Axes | Iterable[plt.Axes], axis: Literal["x", "y", "xy", "yx"]) -> None:
    """Removes ticks from the axis of a figure or axis object.

    If a figure is passed, the function will remove the axis-ticks of all the figure's axes.

    Args
    ----------
    f : Figure | Axes | Iterable[Axes]
        The figure or axis object to remove the ticks from.
    axis : Literal["x", "y", "xy", "yx"]
        The axis to remove the ticks from. If "xy" or "yx" is passed, the function will remove
        the ticks from both axes.

    Raises
    ------
    ValueError
        If f is not a Figure or Axis object.
    """
    match f:
        case plt.Figure():
            axes = f.axes
            [_set_locator_formatter(a, axis) for a in axes]
        case plt.Axes():
            _set_locator_formatter(f, axis)
        case Iterable() | list() | np.ndarray():
            assert all(isinstance(a, plt.Axes) for a in f), "f must be an iterable of Axes objects"
            [_set_locator_formatter(a, axis) for a in f]
        case _:
            raise ValueError("f must be a Figure, an Axes object, or a list of Axes objects")


def _set_table_index(x: pd.Series):
    """Generates a table index string by concatenating the first part of the 'region' column split by '_' and the 'instance_id'.

    Parameters
    ----------
    x : pd.Series
        A pandas Series object representing a row of the table, with at least 'region' and 'instance_id' fields.

    Returns
    -------
    str
        The generated index string for the table.
    """
    return f"{first(x.region.split('_'))}_{x.instance_id}"


def reset_table_index(element: sd.SpatialData | ad.AnnData | pd.DataFrame, table: str | None = None) -> None:
    """Resets the index of the table based on 'region' and 'instance_id' columns.

    This function handles `SpatialData`, `AnnData`, and `DataFrame` objects.

    Parameters
    ----------
    element : sd.SpatialData | ad.AnnData | pd.DataFrame
        The data structure containing the table for which the index needs to be reset.
    table : str | None, optional
        The table name in case of a SpatialData object, by default None.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the DataFrame does not contain the 'region' and 'instance_id' columns.
    """
    match element:
        case sd.SpatialData():
            element.tables[table].obs_names = element.tables[table].obs.apply(_set_table_index, axis=1)
            element.tables[table].strings_to_categoricals()
        case ad.AnnData():
            element.obs_names = element.obs.apply(_set_table_index, axis=1)
            element.strings_to_categoricals()
        case pd.DataFrame():
            if np.isin(["region", "instance_id"], element.columns).all():
                element.index = element.apply(_set_table_index, axis=1)
            else:
                raise ValueError("The DataFrame does not contain the 'region' and 'instance_id' columns.")
    return element


def _extract_layer_from_sdata(
    sdata: sd.SpatialData, vars: Sequence[str] | None, table_name: str, layer: str
) -> ad.AnnData:
    if layer is not None:
        if vars is None:
            adata: ad.AnnData = _convert_layer_to_adata(adata=sdata.tables[table_name].copy(), layer=layer)
        else:
            adata: ad.AnnData = _convert_layer_to_adata(adata=sdata.tables[table_name][:, vars].copy(), layer=layer)
    else:
        if vars is None:
            adata = sdata.tables[table_name]
        else:
            adata = sdata.tables[table_name][:, vars]
    return adata


def _extract_layer_from_sdata(
    sdata: sd.SpatialData, vars: Sequence[str] | None, table_name: str, layer: str
) -> ad.AnnData:
    table: ad.AnnData = sdata.tables[table_name]
    adata: ad.AnnData = table.copy() if vars is None else table[:, vars].copy()
    return _convert_layer_to_adata(adata=adata, layer=layer) if layer is not None else table


def _convert_layer_to_adata(adata: ad.AnnData, layer: str) -> ad.AnnData:
    """Converts a layer in an AnnData object to an AnnData object.

    Parameters
    ----------
    adata
        The AnnData object.
    layer
        The name of the layer to convert.

    Returns
    -------
    A copy of the AnnData object with the layer converted to X.
    """
    layer_as_adata = ad.AnnData(
        X=adata.layers[layer],
        obs=adata.obs,
        var=adata.var,
        uns=adata.uns,
        obsm=adata.obsm,
        varm=adata.varm,
        obsp=adata.obsp,
        varp=adata.varp,
    )

    return layer_as_adata
