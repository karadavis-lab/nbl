from collections.abc import Iterable, Mapping
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd
from matplotlib import ticker


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
    if f"{element_type}/{element_name}" in not_in_zarr_store and not in_zarr_store:
        sdata.write_element(element_name=element_name, overwrite=False)
    else:
        _replace_element(sdata=sdata, element_name=element_name)


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
