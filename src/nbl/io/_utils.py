from collections.abc import Mapping
from typing import Literal

import dask.array as da
import natsort as ns
import xarray as xr
from dask_image.imread import imread
from spatialdata.models import C, Image2DModel, Labels2DModel, X, Y
from spatialdata.transformations import Identity
from upath import UPath


def _rechunk(si: xr.DataArray, chunks: tuple[int, int, int]) -> xr.DataArray:
    """Rechunk a SpatialImage.

    Parameters
    ----------
    si : xr.DataArray
        The SpatialImage to rechunk.
    chunks : tuple[int, int, int]
        The chunks to rechunk the SpatialImage to.

    Returns
    -------
    SpatialImage
        The rechunked SpatialImage.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from nbl.io import _rechunk

        _rechunk(si, chunks=(1, 256, 256))
    """
    si.data = si.data.rechunk(chunks)
    return si


def _parse_image(
    fov_path: UPath, array_type: Literal["numpy", "cupy"], rechunk: tuple[int, int, int] | None
) -> xr.DataArray:
    """Parse a single image from a folder of tiffs.

    Parameters
    ----------
    fov_path : UPath
        Path to the folder containing the tiffs.
    array_type : Literal["numpy", "cupy"]
        The array type to use for the data.
    rechunk : bool | tuple[int, int, int]
        If True, the data will be rechunked to the specified chunk size. If a tuple is provided, it will be used as the chunk size.

    Returns
    -------
    xr.DataArray
        The parsed image.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from nbl.io import _parse_image

        _parse_image(UPath("/path/to/fov/folder"), array_type="numpy", rechunk=(1, 256, 256))

    """
    data: da.Array = imread(fname=f"{fov_path.as_posix()}/*.tiff", arraytype=array_type)
    channels = ns.natsorted([f.stem for f in fov_path.glob("*.tiff")])

    parsed_image: xr.DataArray = Image2DModel().parse(
        data=data, dims=(C, Y, X), c_coords=channels, transformations={fov_path.name: Identity()}
    )
    Image2DModel().validate(parsed_image)
    return _rechunk(parsed_image, chunks=rechunk) if rechunk else parsed_image


def _parse_labels(
    label_path: UPath,
    fov_name: str,
    array_type: Literal["numpy", "cupy"],
    rechunk: tuple[int, int, int] | None,
) -> Mapping[str, xr.DataArray]:
    """Convert segmentation labels for a given FOV to an Xarray DataArray..

    Parameters
    ----------
    label_path : UPath
        The path to the label image.
    fov_name : str
        The name of the FOV.
    array_type : Literal["numpy", "cupy"]
        The array type to use for the data.
    rechunk : tuple[int, int, int] | None
        The chunk size to use for rechunking the data.

    Returns
    -------
    Mapping[str, xr.DataArray]
        A mapping of label names to parsed label images.
    """
    # Get label suffixes
    labels: list[UPath] = ns.natsorted(label_path.glob(f"{fov_name}_*.tiff"))

    labels_map = {}

    for label_path in labels:
        parsed_label = _convert_label_to_labels(
            label_path=label_path, fov_name=fov_name, array_type=array_type, rechunk=rechunk
        )

        label_type = label_path.stem.split(f"{fov_name}_")[1]
        labels_map[f"{fov_name}_{label_type}"] = parsed_label
    return labels_map


def _convert_label_to_labels(
    label_path: UPath,
    fov_name: str,
    array_type: Literal["numpy", "cupy"],
    rechunk: tuple[int, int, int] | None,
) -> xr.DataArray:
    """Convert a single label image to a Labels2DModel / Xarray.

    Parameters
    ----------
    label_path : UPath
        The file path to the label image.
    fov_name : str
        The name of the FOV.
    array_type : Literal["numpy", "cupy"]
        The array type to use for the data.
    rechunk : tuple[int, int, int] | None
        The chunk size to use for rechunking the data.

    Returns
    -------
    xr.DataArray
        The parsed label image.
    """
    label_image = imread(fname=label_path, arraytype=array_type)

    parsed_label = Labels2DModel().parse(
        data=label_image.squeeze() if label_image.ndim == 3 else label_image,
        dims=(Y, X),
        transformations={fov_name: Identity()},
    )
    Labels2DModel().validate(parsed_label)
    if rechunk:
        parsed_label: xr.DataArray = _rechunk(parsed_label, chunks=rechunk)
    return parsed_label
