from dataclasses import dataclass
from typing import NamedTuple, Self

import bionty as bt
import natsort as ns

bt.settings.organism = "human"
cm_lookup: NamedTuple = bt.CellMarker.lookup()


@dataclass
class OntologySet:
    """Represents a set of ontology features for a specific category."""

    name: str
    features: list[bt.CellMarker]
    ontology: bt.models.BioRecord

    @property
    def _featureset_name(self) -> str:
        return f"{self.name} Markers"

    def __repr__(self):
        return self._featureset_name

    def to_list(self) -> list[str]:
        """Get a list of strings representation of the markers in this MarkerSet.

        Returns
        -------
        A list of strings representation of the markers in this MarkerSet.
        """
        return ns.natsorted([marker.name for marker in self.features])

    # def get_featureset(self) -> ln.FeatureSet:
    #     """Retrieve or create the FeatureSet associated with this MarkerSet.

    #     Returns
    #     -------
    #     The FeatureSet associated with this MarkerSet.
    #     """
    #     if not self._check_marker_set_in_db():
    #         self._save_to_db()
    #     return ln.FeatureSet.get(name=self._featureset_name)

    # def _save_to_db(self) -> None:
    #     """Save the FeatureSet to the database if it does not exist."""
    #     fs = ln.FeatureSet(features=self.features, dtype="cat[bionty.CellMarker]", name=self._featureset_name)
    #     fs.save()

    # def _check_marker_set_in_db(self) -> bool:
    #     """Check if the specified marker set exists in the database."""
    #     return ln.FeatureSet.filter(name__exact=self._featureset_name).exists()

    def __iter__(self):
        yield from self.to_list()

    def __add__(self, other: Self) -> Self:
        return OntologySet(
            name=f"{self.name} + {other.name}",
            features=ns.natsorted(set(self.features + other.features), key=lambda x: x.name),
            ontology=self.ontology,
        )


Immune_Infiltrate_Markers = OntologySet(
    name="Immune Infiltrate",
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
    name="Neuroblastoma",
    features=[cm_lookup.th, cm_lookup.gata3, cm_lookup.phox2b, cm_lookup.fn1, cm_lookup.snai2, cm_lookup.vimentin],
    ontology=bt.CellMarker,
)

Neuroblastoma_Extra_Markers = OntologySet(
    name="Neuroblastoma Extra",
    features=[
        cm_lookup.trka,
        cm_lookup.trkb,
        cm_lookup.synaptophysin,
    ],
    ontology=bt.CellMarker,
)

Adrenergic_Markers = OntologySet(
    name="Adrenergic",
    features=[
        cm_lookup.th,
        cm_lookup.gata3,
        cm_lookup.phox2b,
    ],
    ontology=bt.CellMarker,
)

Mesenchymal_Markers = OntologySet(
    name="Mesenchymal",
    features=[cm_lookup.fn1, cm_lookup.snai2, cm_lookup.vimentin],
    ontology=bt.CellMarker,
)

Stem_Cell_Markers = OntologySet(
    name="Stem Cell",
    features=[
        cm_lookup.cd117,
    ],
    ontology=bt.CellMarker,
)

Intracellular_Markers = OntologySet(
    name="Intracellular",
    features=[cm_lookup.sox10, cm_lookup.ki67, cm_lookup.ps6],
    ontology=bt.CellMarker,
)

Tissue_Structure_Markers = OntologySet(
    name="Tissue Structure",
    features=[cm_lookup.hh3dsdna, cm_lookup.sma, cm_lookup.cd31],
    ontology=bt.CellMarker,
)

Cell_Surface_Markers = OntologySet(
    name="Cell Surface",
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

All_Markers = OntologySet(
    name="All",
    features=[
        # Immune Infiltrate Markers
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
        # Neuroblastoma Markers (including Extra)
        cm_lookup.th,
        cm_lookup.gata3,
        cm_lookup.phox2b,
        cm_lookup.fn1,
        cm_lookup.snai2,
        cm_lookup.vimentin,
        cm_lookup.trka,
        cm_lookup.trkb,
        cm_lookup.synaptophysin,
        # Stem Cell Markers
        cm_lookup.cd117,
        # Intracellular Markers
        cm_lookup.sox10,
        cm_lookup.ki67,
        cm_lookup.ps6,
        # Tissue Structure Markers
        cm_lookup.hh3dsdna,
        cm_lookup.sma,
        cm_lookup.cd31,
        # Cell Surface Markers (unique ones not already included)
        cm_lookup.b7_h3,
        cm_lookup.gpc2,
        cm_lookup.ncaml1,
    ],
    ontology=bt.CellMarker,
)
