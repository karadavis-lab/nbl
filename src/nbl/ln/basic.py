from typing import Literal

import lamindb as ln

from .featuresets import cell_marker_set_map


class CellMarkerSetCatalog:
    """Catalog of multiple cell marker feature sets."""

    def __init__(self, cell_marker_set: str, return_type: Literal["featureset", "names"]) -> ln.FeatureSet | list[str]:
        """Returns the FeatureSet or a list of marker names, depending on `return_type` for a given cell marker set.

        Parameters
        ----------
        cell_marker_set : str
            The name of the cell marker set to retrieve.
        return_type : Literal["featureset", "names"]
            The format of the returned data:
            - "featureset": returns the FeatureSet object
            - "names": returns a list of marker names

        Returns
        -------
        ln.FeatureSet | list[str]
            The FeatureSet or a list of marker names, depending on `return_type`.

        Raises
        ------
        ValueError
            If the specified cell marker set is not valid.

        Examples
        --------
        >>> cmc = CellMarkerSetCatalog(cell_marker_set="immune_infiltrate", return_type="featureset")
        """
        self._cell_marker_set_map = cell_marker_set_map
        self._return_type = return_type

        # simple test to validate param value
        if cell_marker_set in self._cell_marker_set_map.keys():
            self.cell_marker_set = cell_marker_set
        else:
            raise ValueError(f"Invalid Value for cell_marker_set: {cell_marker_set}")
        self.__call__()

    def __call__(self) -> ln.FeatureSet | list[str]:
        """Returns the FeatureSet or a list of marker names, depending on `return_type`."""
        return self._cell_marker_set_map[self.cell_marker_set].get_markers(self._return_type)


def cell_marker_set_catalog(
    cell_marker_set: str, return_type: Literal["featureset", "names"]
) -> ln.FeatureSet | list[str]:
    """Returns the FeatureSet or a list of marker names, depending on `return_type` for a given cell marker set.

    Parameters
    ----------
    cell_marker_set : str
        The name of the cell marker set to retrieve.
    return_type : Literal["featureset", "names"]
        The format of the returned data:
        - "featureset": returns the FeatureSet object
        - "names": returns a list of marker names

    Returns
    -------
    ln.FeatureSet | list[str]
        The FeatureSet or a list of marker names, depending on `return_type`.

    Raises
    ------
    ValueError
        If the specified cell marker set is not valid.

    Examples
    --------
    >>> cmc = CellMarkerSetCatalog(cell_marker_set="immune_infiltrate", return_type="featureset")
    """
    return CellMarkerSetCatalog(cell_marker_set=cell_marker_set, return_type=return_type)()
