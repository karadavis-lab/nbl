import anndata as ad
import numpy as np
import spatialdata as sd
from numpy.typing import NDArray
from scipy import sparse

from nbl._util import write_elements


def arcsinh_transform(
    sdata: sd.SpatialData,
    table_names: str | list[str] = None,
    shift_factor: np.integer | np.floating = 0,
    scale_factor: np.integer | np.floating = 5,
    replace_X: bool = False,
    write: bool = True,
    inplace: bool = True,
) -> None | sd.SpatialData:
    r"""Apply the arcsinh transformation to the data in the specified tables of the SpatialData object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The SpatialData object containing the data.
    table_names : str | list[str], optional
        The name(s) of the table(s) to transform. If None, all tables will be transformed, by default None.
    shift_factor : np.integer | np.floating, optional
        The shift factor to be added to the data before applying the transformation, by default 0
    scale_factor : np.integer | np.floating, optional
        The scale factor to be multiplied with the data after applying the transformation, by default 5
    replace_X : bool, optional
        Whether to replace the original data in `AnnData.X` with the transformed data, by default False
    write : bool, optional
        Whether to write the transformed tables back to the SpatialData object's Zarr store, by default True
    inplace : bool, optional
        Whether to modify the SpatialData object in-place or return a new object, by default True

    Returns
    -------
    None | sd.SpatialData
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

        if replace_X:
            adata.X = sparse.csr_matrix(transformed_X)
        else:
            adata.layers[f"arcsinh_shift_{shift_factor}_scale_{scale_factor}"] = sparse.csr_matrix(transformed_X)

    if write:
        write_elements(sdata=sdata, elements={"tables": table_keys})
    return None if inplace else sdata
