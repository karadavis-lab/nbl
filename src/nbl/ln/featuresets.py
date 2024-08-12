from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NamedTuple

import bionty as bt
import lamindb as ln
import natsort as ns
from bionty.models import BioRecord
from lamindb.core import QuerySet, Record

bt.settings.organism = "human"
cm_lookup: NamedTuple = bt.CellMarker.lookup()

__all__ = ["CellMarkerSet"]


class OntologySet(NamedTuple):
    name: str
    features: list[str]
    ontology: BioRecord


Immune_Infiltrate_Markers = OntologySet(
    name="Immune Infiltrate Markers",
    features=[
        cm_lookup.calprotectin,
        cm_lookup.cd3,
        cm_lookup.cd4,
        cm_lookup.cd8,
        cm_lookup.cd11b,
        cm_lookup.cd11c,
        cm_lookup.cd14,
        cm_lookup.cd15,
        cm_lookup.cd16,
        cm_lookup.cd20,
        cm_lookup.cd45,
        cm_lookup.cd56,
        cm_lookup.cd57,
        cm_lookup.cd68,
        cm_lookup.cd166,
        cm_lookup.cd206,
        cm_lookup.cd209,
        cm_lookup.pd1,
        cm_lookup.pdl1,
        cm_lookup.hla_class_i,
    ],
    ontology=bt.CellMarker,
)

Neuroblastoma_Markers = OntologySet(
    name="Neuroblastoma Markers",
    features=[cm_lookup.th, cm_lookup.gata3, cm_lookup.phox2b, cm_lookup.fn1, cm_lookup.snai2, cm_lookup.vimentin],
    ontology=bt.CellMarker,
)

Adrenergic_Markers = OntologySet(
    name="Adrenergic Markers",
    features=[
        cm_lookup.th,
        cm_lookup.gata3,
        cm_lookup.phox2b,
    ],
    ontology=bt.CellMarker,
)

Mesenchymal_Markers = OntologySet(
    name="Mesenchymal Markers",
    features=[cm_lookup.fn1, cm_lookup.snai2, cm_lookup.vimentin],
    ontology=bt.CellMarker,
)

Stem_Cell_Markers = OntologySet(
    name="Stem Cell Markers",
    features=[
        cm_lookup.cd117,
    ],
    ontology=bt.CellMarker,
)

Intracellular_Markers = OntologySet(
    name="Intracellular Markers",
    features=[cm_lookup.sox10, cm_lookup.ki67, cm_lookup.ps6],
    ontology=bt.CellMarker,
)

Tissue_Structure_Markers = OntologySet(
    name="Tissue Structure Markers",
    features=[cm_lookup.hh3dsdna, cm_lookup.sma, cm_lookup.cd31],
    ontology=bt.CellMarker,
)

Cell_Surface_Markers = OntologySet(
    name="Cell Surface Markers",
    features=[
        cm_lookup.b7_h3,
        cm_lookup.cd56,
        cm_lookup.cd166,
        cm_lookup.gpc2,
        cm_lookup.ncaml1,
        cm_lookup.trka,
        cm_lookup.trkb,
        cm_lookup.synaptophysin,
    ],
    ontology=bt.CellMarker,
)


@dataclass(init=False)
class CellMarkerSet:
    """Class to handle operations related to cell markers.

    Returns
    -------
    CellMarkerSet
        A CellMarkerSet object
    """

    name: str
    features: list[Record]
    ontology: BioRecord
    featureset: ln.FeatureSet

    def __init__(self, ontology_set: OntologySet):
        self.features = ontology_set.features
        self.name = ontology_set.name
        self.ontology = ontology_set.ontology

    def _save_to_db(self) -> None:
        """Save the FeatureSet to the database."""
        fs = ln.FeatureSet(features=self.features, dtype="cat[bionty.CellMarker]", name=self.name)
        fs.save()

    @cached_property
    def _check_marker_set_in_db(self) -> bool:
        """Check if the specified marker set exists in the database.

        Returns
        -------
        bool
            True if the marker set exists, False otherwise
        """
        return ln.FeatureSet.filter(name__exact=self.name).exists()

    @cached_property
    def _featureset(self) -> ln.FeatureSet:
        """Get the FeatureSet from the database."""
        return ln.FeatureSet.filter(name__exact=self.name).one()

    @cached_property
    def _featureset_members(self) -> list[str]:
        return ns.natsorted([im.name for im in self._featureset.members])

    def get_markers(self, return_type: Literal["featureset", "names"]) -> QuerySet | list[str]:
        """Get markers and ensure they are saved to the database if not already.

        Parameters
        ----------
        return_type : Literal["featureset", "names"]
            Determines the format of the returned data:
            - "featureset": returns the FeatureSet object
            - "names": returns a list of marker names

        Returns
        -------
        QuerySet | list[str]
            The FeatureSet or a list of marker names, depending on `return_type`
        """
        if not self._check_marker_set_in_db:
            self._save_to_db()
        match return_type:
            case "featureset":
                return self._featureset
            case "names":
                return self._featureset_members


cell_marker_set_map: Mapping[str, CellMarkerSet] = {
    "immune_infiltrate": CellMarkerSet(ontology_set=Immune_Infiltrate_Markers),
    "neuroblastoma": CellMarkerSet(ontology_set=Neuroblastoma_Markers),
    "adrenergic": CellMarkerSet(ontology_set=Adrenergic_Markers),
    "mesenchymal": CellMarkerSet(ontology_set=Mesenchymal_Markers),
    "stem_cell": CellMarkerSet(ontology_set=Stem_Cell_Markers),
    "intracellular": CellMarkerSet(ontology_set=Intracellular_Markers),
    "tissue_structure": CellMarkerSet(ontology_set=Tissue_Structure_Markers),
    "cell_surface": CellMarkerSet(ontology_set=Cell_Surface_Markers),
}
"""Mapping of cell marker sets to their corresponding FeatureSet objects."""
