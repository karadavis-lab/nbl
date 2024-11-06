import re
from typing import Literal

import natsort as ns
import spatialdata as sd
from dask import delayed
from upath import UPath

from nbl.util import DaskSetupDelayed

from ._utils import _parse_image, _parse_labels


def convert_cohort(
    fov_dir: UPath,
    label_dir: UPath,
    file_path: UPath,
    filter_fovs: str | re.Pattern = None,
    array_type: Literal["numpy", "cupy"] = "numpy",
    return_sdata: bool = False,
) -> sd.SpatialData | None:
    """Converts a cohort of images and labels to SpatialData objects and saves them to disk.

    Parameters
    ----------
    fov_dir
        The directory containing the field of view images.
    label_di
        The directory containing the label images.
    filter_fovs:
        A regular expression pattern to filter the FOVs by.
    file_path
        The path to the Zarr Store to save the `SpatialData` objects to.
    array_type
        Array type for dask chunks. Available options: "numpy", "cupy".
    return_sdata
        Whether to return the `SpatialData` object or not.

    Returns
    -------
    sd.SpatialData | None
        The `SpatialData` object if `return_sdata` is True, otherwise None.

    """
    fovs = ns.natsorted(fov_dir.glob("[!.]*/"))
    if filter_fovs:
        re_pattern = re.compile(pattern=filter_fovs)
        fovs = filter(lambda p: re_pattern.search(string=p.as_posix()), fovs)

    _tasks = []
    for fov in fovs:
        _tasks.append(delayed(_create_sdata)(fov, label_dir, array_type))

    dask_runner = DaskSetupDelayed(delayed_objects=_tasks)
    _sdatas = dask_runner.compute()
    sdata = sd.concatenate(sdatas=_sdatas)

    sdata.write(file_path=file_path)
    if return_sdata:
        return sdata


def _create_sdata(
    fov_path: UPath,
    label_dir: UPath,
    array_type: Literal["numpy", "cupy"] = "numpy",
    rechunk: bool | tuple[int, int, int] = False,
) -> sd.SpatialData:
    """
    Creates a SpatialData object from field of view (FOV) images and label data.

    Parameters
    ----------
    fov_path : UPath
        Path to the field of view (FOV) image file.
    label_dir : UPath
        Directory path containing label files.
    array_type : Literal["numpy", "cupy"], optional
        The type of array to use for image and label data. Can be "numpy" or "cupy". Default is "numpy".
    rechunk : bool | tuple[int, int, int], optional
        Option to rechunk the array data. If a tuple of three integers is provided, it represents
        the chunk size. Default is False.

    Returns
    -------
    sd.SpatialData
        A SpatialData object containing the FOV image and labeled data.

    Notes
    -----
    This function initializes an empty SpatialData object and populates it with parsed image and label data.
    The parsing functions `_parse_image` and `_parse_labels` are assumed to handle the respective file formats
    and convert them to the appropriate array types.
    """
    fov_name = fov_path.name

    sdata = sd.SpatialData()

    parsed_image = _parse_image(fov_path, array_type, rechunk)

    parsed_labels = _parse_labels(label_path=label_dir, fov_name=fov_name, array_type=array_type, rechunk=rechunk)

    sdata.images[fov_name] = parsed_image
    for label_name, parsed_label in parsed_labels.items():
        sdata.labels[label_name] = parsed_label
    return sdata
