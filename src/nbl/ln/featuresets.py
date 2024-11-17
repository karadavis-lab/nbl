from dataclasses import dataclass
from typing import NamedTuple

import bionty as bt
import lamindb as ln
import natsort as ns

bt.settings.organism = "human"
cm_lookup: NamedTuple = bt.CellMarker.lookup()


@dataclass
class OntologySet:
    """Represents a set of ontology features for a specific category."""

    name: str
    features: list[bt.CellMarker]
    ontology: bt.models.BioRecord

    def __repr__(self):
        return self.name

    def to_list(self) -> list[str]:
        """Get a list of strings representation of the markers in this MarkerSet.

        Returns
        -------
        A list of strings representation of the markers in this MarkerSet.
        """
        return ns.natsorted([marker.name for marker in self.get_featureset().members])

    def get_featureset(self) -> ln.FeatureSet:
        """Retrieve or create the FeatureSet associated with this MarkerSet.

        Returns
        -------
        The FeatureSet associated with this MarkerSet.
        """
        fs_name = self.name
        if not self._check_marker_set_in_db():
            self._save_to_db()
        return ln.FeatureSet.filter(name__exact=fs_name).one()

    def _save_to_db(self) -> None:
        """Save the FeatureSet to the database if it does not exist."""
        fs = ln.FeatureSet(features=self.features, dtype="cat[bionty.CellMarker]", name=self.name)
        fs.save()

    def _check_marker_set_in_db(self) -> bool:
        """Check if the specified marker set exists in the database."""
        return ln.FeatureSet.filter(name__exact=self.name).exists()

    def __iter__(self):
        yield from self.to_list()


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

Neuroblastoma_Markers_Extra = OntologySet(
    name="Neuroblastoma Markers Extra",
    features=[
        cm_lookup.trka,
        cm_lookup.trkb,
        cm_lookup.synaptophysin,
    ],
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
