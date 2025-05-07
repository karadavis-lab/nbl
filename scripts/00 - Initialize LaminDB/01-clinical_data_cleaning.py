import re
from pathlib import Path
from typing import Annotated, Any

import bionty as bt
import lamindb as ln
import more_itertools as mit
import natsort as ns
import pandas as pd
import rapidfuzz as rfuzz
import typer
from upath import UPath

import nbl
from nbl.ln.schemas import clinical_schema

bt.settings.organism = "human"
logger = nbl.logger
app = typer.Typer()


def load_data(clinical_data_path: UPath) -> pd.DataFrame:
    """Load clinical data from Excel file.

    Args:
        clinical_data_path: Path to clinical data Excel file

    Returns
    -------
        Loaded DataFrame
    """
    logger.info(f"Loading clinical data from {clinical_data_path}")
    return pd.read_excel(clinical_data_path)


def standardize_fovs(clinical_data: pd.DataFrame, fov_dir: UPath) -> pd.DataFrame:
    """Standardize Field of View names/IDs.

    Args:
        clinical_data: Clinical data DataFrame
        fov_dir: Directory containing FOV images

    Returns
    -------
        DataFrame with standardized FOV names
    """
    logger.info("Standardizing FOV names")
    all_fovs = ns.natsorted(fov_dir.glob("[!.]*/"))
    control_pattern = re.compile(pattern=r"Hu-*")
    all_fov_names = [fov.name for fov in all_fovs]
    fov_names = list(filter(lambda f: not control_pattern.search(f), all_fov_names))

    def convert_fov(row, fovs: list[str]) -> str | None:
        for fov in fovs:
            if row["fov"] == fov.split("-")[2]:
                return fov
        return None

    clinical_data["fov"] = clinical_data.apply(lambda row: convert_fov(row, fov_names), axis=1)  # type: ignore
    return clinical_data


def clean_column_names(clinical_data: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing whitespace.

    Args:
        clinical_data: Clinical data DataFrame

    Returns
    -------
        DataFrame with cleaned column names
    """
    logger.info("Cleaning column names")
    clinical_data.columns = clinical_data.columns.str.strip()
    return clinical_data


def clean_paired_sequence(clinical_data: pd.DataFrame) -> pd.DataFrame:
    """Clean paired sequence column.

    Args:
        clinical_data: Clinical data DataFrame

    Returns
    -------
        DataFrame with cleaned paired sequence
    """
    logger.info("Cleaning paired sequence")
    clinical_data["Paired sequence"] = clinical_data["Paired sequence"].map(lambda x: False if x == "No" else True)
    return clinical_data


def clean_data_values(clinical_data: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize data values in the DataFrame.

    Args:
        clinical_data: Clinical data DataFrame

    Returns
    -------
        DataFrame with cleaned values
    """
    logger.info("Cleaning data values")

    # Get lookup tables
    tissues = bt.Tissue.public()
    ethnicities = bt.Ethnicity.public()
    tissues_lookup = tissues.lookup()
    ethnicity_lookup = ethnicities.lookup()

    # Define replacements
    replacements = {
        "Classification of specimen": {
            "Diagnosis": "Diagnosis",
            "post-chemotherapy, local control surgery ": "Post-Chemotherapy",
            "Diagnosis ": "Diagnosis",
            "post-chemotherpy, local control surgery (mild paraspinal disease progression requiring laminectomy)": "Post-Chemotherapy",
            "post-chemotherapy (local control surgery, 4 cycles of ANBL0531) ": "Post-Chemotherapy",
            "relapse (after 2 cycles of topo/cyclo)": "Relapse",
            "Progressive disease (re-resection, s/p chemotherapy) ": "Disease Progression",
            "post-chemotherapy, local control surgery (s/p 4 cycles of induction chemo per ANBL0531) ": "Post-Chemotherapy",
            "post-chemotherapy (5 cycles ANBL0532) ": "Post-Chemotherapy",
            "Relapsed": "Relapse",
            "CCHS, post-chemo therapy, local control surgery (7 cycles of ANBL0531, stable disease after 6 cycles and then 1 cycle of topo/cyclo) ": "Post-Chemotherapy",
            "relapse, brain metastases": "Relapse",
            "post-chemotherapy, local control surgery (2nd)": "Post-Chemotherapy",
            "post-chemotherapy, local control surgery (s/p 4 cycles of induction per ANBL0531) ": "Post-Chemotherapy",
            "post-chemotherapy, local control surgery (8 cyles of ANBL0531 therapy with minimal response)  ": "Post-Chemotherapy",
            "Diagnosis (after a period of observation) ": "Diagnosis",
            "disease progression after upfront surgery (posterior mediastinum)": "Disease Progression",
            "post-chemotherapy, local control surgery": "Post-Chemotherapy",
        },
        "Sex": {s: s.strip().lower().capitalize() for s in clinical_data["Sex"].unique()},
        "Race": {
            "Black": ethnicity_lookup.african.name,
            "White": ethnicity_lookup.european.name,
            "white": ethnicity_lookup.european.name,
            "Other": ethnicity_lookup.undefined_ancestry_population.name,
            "Arabic ": ethnicity_lookup.arab.name,
            "Asian ": ethnicity_lookup.asian.name,
            "other (egyptian)": ethnicity_lookup.egyptian.name,
            "?black ": ethnicity_lookup.african.name,
            "white ": ethnicity_lookup.european.name,
        },
        "Biopsy/surgery location": {
            "abdominal mass": tissues_lookup.abdominal_segment_element.name,
            "letfy adrenal mass": tissues_lookup.left_adrenal_gland.name,
            "Right adrenal ": tissues_lookup.right_adrenal_gland.name,
            "Abdominal mass": tissues_lookup.abdominal_segment_element.name,
            "Spinal/paraspinal ": tissues_lookup.paraspinal_region.name,
            "RP mass ": tissues_lookup.retroperitoneal_space.name,
            "abdominal mass/thoracic region mass excision": tissues_lookup.thoracic_cavity_element.name,
            "abdominal mass/diagphramtic mass": tissues_lookup.diaphragm.name,
            "left adrenal tumor": tissues_lookup.left_adrenal_gland.name,
            "pelvic mass": tissues_lookup.pelvic_region_element.name,
            "abdominal tumor resection ": tissues_lookup.abdominal_segment_element.name,
            "Retroperitoneal": tissues_lookup.retroperitoneal_space.name,
            "Abdominal/Retroperitoneal": tissues_lookup.retroperitoneal_space.name,
            "Pelvic mass, s/p 2 cycles of ANBL0531, limited response to chemo with tumor growth": tissues_lookup.pelvic_region_element.name,
            "Paraspinal ": tissues_lookup.paraspinal_region.name,
            "paraspinal ": tissues_lookup.paraspinal_region.name,
            "Right Adrenal": tissues_lookup.right_adrenal_gland.name,
            "Liver": tissues_lookup.liver.name,
            "abdominal tumor": tissues_lookup.abdominal_segment_element.name,
            "Abdominal tumor, lymph nodes": tissues_lookup.abdominal_lymph_node.name,
            "Brain mets, relapse during maintenance GD2 antibody": tissues_lookup.brain.name,
            "paraspinal mass": tissues_lookup.paraspinal_region.name,
            "abdominal tumor resection": tissues_lookup.abdominal_segment_element.name,
            "right adrenal mass": tissues_lookup.right_adrenal_gland.name,
            "right adrenal gland resection ": tissues_lookup.right_adrenal_gland.name,
            "retroperitoneal mass": tissues_lookup.retroperitoneal_space.name,
            "neck mass": tissues_lookup.neck.name,
            "abdominal/paraspinal mass resection": tissues_lookup.paraspinal_region.name,
            "right chect, posterior mediastinal ": tissues_lookup.posterior_mediastinum.name,
            "retroperitoneal": tissues_lookup.retroperitoneal_space.name,
            "abdominal tumor resection after 4 cycles of ANBL0531 ": tissues_lookup.abdominal_segment_element.name,
            "right adrenal gland": tissues_lookup.right_adrenal_gland.name,
            "right apical chest mass resection": tissues_lookup.chest.name,
            "abdominal mass/liver nodule": tissues_lookup.liver.name,
            "pelvic tumor": tissues_lookup.pelvic_region_element.name,
            "right neck mass": tissues_lookup.neck.name,
            "Abd mass": tissues_lookup.abdominal_segment_element.name,
            "abdominal mass biopsy": tissues_lookup.abdominal_segment_element.name,
            "right axilla": tissues_lookup.axilla.name,
            "thoracic tumor": tissues_lookup.thoracic_cavity_element.name,
            "left chest mass": tissues_lookup.chest.name,
            "b/l adrenal masses": tissues_lookup.adrenal_tissue.name,
            "adrenalectomy": tissues_lookup.adrenal_tissue.name,
            "lefty adrenal mass": tissues_lookup.left_adrenal_gland.name,
        },
        "Risk": {
            "Intermediate": "Intermediate",
            "High": "High",
            "Inrtermediate, mild disease progression": "Intermediate",
            "intermediate": "Intermediate",
            "High (relapsed)": "High",
            "Intermediate ": "Intermediate",
            "Low": "Low",
            "High ": "High",
            "Low (would be IR now?)": "Low",
            "High (due to nodular ganglioneuroblastoma)": "High",
            "Intermediate  ": "Intermediate",
        },
    }

    # Apply replacements
    clinical_data = clinical_data.replace(to_replace=replacements)

    # Clean HVA/VMA data
    u_vma_hva = "Urine VMA/HVA (g/g Cr)"
    clinical_data = clinical_data.replace(to_replace={u_vma_hva: {"n/a ": pd.NA, ">227/>227": "227/227"}})
    clinical_data = clinical_data.fillna({u_vma_hva: pd.NA})

    # Split HVA/VMA into separate columns
    _vma_hva_df = (
        clinical_data[u_vma_hva]
        .str.split("/", expand=True)
        .rename(columns={0: "VMA (g Cr)", 1: "HVA (g Cr)"})
        .apply(pd.to_numeric, errors="coerce")
    )
    clinical_data = clinical_data.drop(columns=[u_vma_hva])

    # Clean genomic data
    for c in ["17q gain", "11q loss/LOH", "7q gain", "1p loss/LOH", "ALK"]:
        clinical_data[c] = clinical_data[c].str.rstrip().str.lstrip().str.capitalize()

    # Clean genomic data values
    genomic_replacements = {
        "17q gain": {
            "Yes  (wc, relative, 4n)": "Yes|WC|relative|4N",
            "Yes  (relative, 5n)": "Yes|relative|5N",
            "Yes , (relative, (5n)": "Yes|relative|5N",
            "Yes, relative, 4n)": "Yes|relative|4N",
            "Yes (wc, relative 5n)": "Yes|WC|relative|5N",
            "Yes (wc, relative 4n)": "Yes|WC|relative|4N",
            "Yes (relative wc, 4-5n)": "Yes|WC|relative|4N|5N",
            "Yes (relative, wc, 6n)": "Yes|WC|relative|6N",
            "Yes, 4n (relative)": "Yes|relative|4N",
            "Yes, wc, relative, 4n)": "Yes|WC|relative|4N",
            "Yes (6n)": "Yes|6N",
            "Yes (wc, relative, 4n)": "Yes|WC|relative|4N",
            "Yes (wc, relatve, 4n)": "Yes|WC|relative|4N",
            "Yes (relative, 4n)": "Yes|relative|4N",
            "Yes (wc relative gain)": "Yes|WC|relative|gain",
            "Yes (wc, 4n)": "Yes|WC|4N",
            "Yes 9wc, relative, 4n)": "Yes|WC|relative|4N",
            "Yes (wc, releative, 4n)": "Yes|WC|relative|4N",
        },
        "7q gain": {
            "No ": "No",
            "Yes ": "Yes",
            "Yes (wc)": "Yes|WC",
            "yes (wc, relative, 4n)": "Yes|WC|relative|4N)",
            "Yes": "Yes",
            "Yes (relative, 6n)": "Yes|relative|6N",
            "No": "No",
            "Yes (wc, relagtive 5n)": "Yes|WC|relative|5N",
            "Yes (relative, 4-5n)": "Yes|relative|4N|5N|",
            "Yes (wc, relative, 4n)": "Yes|WC|relative|4N",
            "Yes, wc, relative, 4n)": "Yes|WC|relative|4N",
            "Yes (wc, 4n)": "Yes|WC|4N",
            "Yes (relative, 4n)": "Yes|relative|4N",
            "Yes 9wc, relative, 4Nn": "Yes|WC|relative|4N",
            "Yes (wc, releative, 4n)": "Yes|WC|relative|4N",
            "Yes (wc, relative, 4n) ": "Yes|WC|relative|4N",
            "Yes (wc relative gain)": "Yes|WC|relative|gain",
            "Yes 9wc, relative, 4n)": "Yes|WC|relative|4N",
        },
        "1p loss/LOH": {
            "No": "No",
            "Yes (relative, 2n)": "Yes|relative|2N",
            "Yes (relative, 2n, cnloh)": "Yes|relative|2N|cnLOH",
            "Yes": "Yes",
            "Yes (wc)": "Yes|WC",
            "No?": "No",
        },
        "11q loss/LOH": {
            "No": "No",
            "Yes": "Yes",
            "Yes, (cn neutral loh)": "Yes|neutral cnLOH",
            "Yes (relative, 2n, cnloh)": "Yes|relative|2N|cnLOH",
            "Yes (deletion)": "Yes|deletion",
            "Yes (relative, 2n, wc, cn loh)": "Yes|relative|WC|2N|cnLOH",
            "Yes (wc)": "Yes|WC",
            "Yes (relative, wc, 2n)": "Yes|relative|WC|2N",
            "Yes (wc, relative loss)": "Yes|relative|WC",
            "Yes (cn neutral loh)": "Yes|neutral|cnLOH",
        },
        "ALK": {
            "Wt": "WT",
            "F1245l (somatic)": "F1245L|somatic",
            "Wt (alk gain)": "WT|ALK gain",
            "Wt (phox2b wt)": "WT|Phox2B WT",
            "Wt/phox2b with a heterozygous polyalanine expansion (20/33).": "WT|Phox2B with a heterozygous polyalanine expansion (20/33)",
            "N/a": pd.NA,
            "Arg1275gln": "Arg1275Gln",
            "Wt (diagnosis and this specimen)": "WT",
            "wt": "WT",
            "WT (PHOX2b WT)": "WT|Phox2B WT",
            "Wt / phox2b wt": "WT|Phox2B WT",
            "F1174l": "F1174L",
        },
    }
    clinical_data = clinical_data.replace(to_replace=genomic_replacements)

    # Clean clinical data values
    clinical_replacements = {
        "INSS stage": {"4s": "4S", "2a": "2A", "2b": "2B", "2b??": "2B"},
        "INRG stage": {
            "M": "M",
            "L2": "L2",
            "Ms": "MS",
            "M (from diagnosis)": "M",
            "L1": "L1",
        },
        "Ploidy value": {
            "Hyperdiplod (3-4n)": "Hyperdiploid|3N|4N",
            "Diploid": "Diploid",
            "Diploid (diagnosis)": "Diploid",
            "Hyperdiploid (3-4n)": "Hyperdiploid",
            "Hyeperdiplod (3n)": "Hyperdiploid|3N",
            "Hyperdiploid (3n)": "Hyperdiploid|3N",
            "Hyperdiploid (3-4n) with scas": "Hyperdiploid|3N|4N",
            "Hyperdiploid (3n of 10 chromosomes)": "Hyperdiploid|3N",
            "Hyperdipoid (3n)": "Hyperdiploid|3N",
            "Hyperdiploid (3n) w/ scas": "Hyperdiploid|3N",
            "Diploid (hyperdiploid at diagnosis?)": "Diploid",
            "Hyperdip)loid (3n)": "Hyperdiploid|3N",
            "Hyperidiploid/with scas": "Hyperdiploid",
            "Hyperploid (3n)": "Hyperdiploid|3N",
            "Hyperdiploid": "Hyperdiploid",
            "Hyperdiploid (near 4n)": "Hyperdiploid|4N",
            "Hyeperdiplid (3n)": "Hyperdiploid|3N",
        },
        "MKI": {
            "Intermediate": "Intermediate",
            "Low": "Low",
            "High": "High",
            "High (diagnostic)": "High",
            "Low (<1%, diagnostic)": "Low",
            "Intermediate (diagnosis)": "Intermediate",
            "Low/intermediate": "Intermediate",
            "Diagnosis = low": "Low",
            "High (from diagnosis)": "High",
            "Low (diagnosis)": "Low",
            "Low (diagnostic)": "Low",
            "High (and one clone with low)": "High",
        },
        "Histologic classification - INPC": {
            "Favorable histology": "Favorable",
            "Favorbale histology, diagnosis = favorable histology": "Favorable",
            "Unfavorable histology": "Unfavorable",
            "Unfavorable histology (diagnosis)": "Unfavorable",
            "Favorable (diagnosis)": "Favorable",
            "Diagnosis = unfavorable histology": "Unfavorable",
            "Unfavorable histology (from diagnosis)": "Unfavorable",
            "Favorable histology (diagnosis)": "Favorable",
            "Favorable histology (diagnostic)": "Favorable",
            "N/a": pd.NA,
            "Unfavorable histology (diagnostic tumor)": "Unfavorable",
            "Unfavorable histology = diagnosis": "Unfavorable",
            "Unfavorable histology (nodular gangloneuroblastoma with a poorly differentiated neuroblastic component)": "Unfavorable",
            "Diagnosis = favorable histology": "Favorable",
            "Favorable (diagnostic)": "Favorable",
            "Unfavorable": "Unfavorable",
            "Unfavorable hiostology (diagnostic)": "Unfavorable",
            "Favorable  (diagnosis)": "Favorable",
            "Favorbale histology": "Unfavorable",
        },
    }
    clinical_data = clinical_data.replace(to_replace=clinical_replacements)

    # Clean degree of differentiation using rapidfuzz
    clinical_data["Degree of differentiation"] = clinical_data["Degree of differentiation"].apply(map_rapidfuzz)

    # Clean MYCN amplification
    clinical_data["MYCN amplification"] = clinical_data["MYCN amplification"].map(
        lambda x: False if "No" in x else True
    )

    # Convert columns to categorical
    categorical_columns = [
        "Classification of specimen",
        "Risk",
        "Biopsy/surgery location",
        "Sex",
        "Race",
        "17q gain",
        "11q loss/LOH",
        "7q gain",
        "1p loss/LOH",
        "ALK",
        "Other mutations (source)",
        "Genomic studies done",
        "INSS stage",
        "INRG stage",
        "Ploidy value",
        "MKI",
        "Degree of differentiation",
        "Histologic classification - INPC",
        "Genomics source",
        "UID",
        "FOV",
    ]
    clinical_data[categorical_columns] = clinical_data[categorical_columns].astype("category")

    # Rename columns
    column_renames = {
        "Race": "Ethnicity",
        "Biopsy/surgery location": "Tissue",
        "11q loss/LOH": "11q LOH",
        "1p loss/LOH": "1p LOH",
        "Classification of specimen": "Classification",
        "fov": "FOV",
        "Age (days) at time of diagnosis (relapse)": "Age (days) at Diagnosis",
        "Age (days) at time of biospy": "Age (days) at Biopsy",
        "HVA/VMA days from biospy": "HVA/VMA (days) from biopsy",
    }
    clinical_data = clinical_data.rename(columns=column_renames)

    # Fill NA values
    clinical_data = clinical_data.fillna(value=pd.NA)

    return clinical_data


def map_rapidfuzz(text: Any) -> str | None:
    """Map text to a choice using rapidfuzz.

    Args:
        text: Text to map

    Returns
    -------
        Mapped choice or None
    """
    if not isinstance(text, str):
        return None

    choices = ["Poorly Differentiated", "Differentiating", "Undifferentiated"]
    result = rfuzz.process.extractOne(text, choices, scorer=rfuzz.fuzz.partial_ratio)

    if result and result[1] > 80:  # Apply confidence threshold
        return result[0]
    else:
        logger.warning(f"Low rapidfuzz match score for: {text} -> {result}")
        return None


def create_sample_schema() -> ln.Schema:
    """Create LaminDB schema for clinical data.

    Returns
    -------
        Created schema
    """
    logger.info("Creating clinical data schema")
    return clinical_schema


def validate_markers(fov_dir: UPath) -> None:
    """Validate cell markers in FOV images.

    Args:
        fov_dir: Directory containing FOV images
    """
    logger.info("Validating cell markers")
    sample_fov_markers = set(ns.natsorted(m.stem for m in fov_dir.glob("*/*.tiff")))
    cell_markers = bt.CellMarker.public()

    _inspected_markers = cell_markers.inspect(values=sample_fov_markers, field=cell_markers.name)
    standardized_markers_mapper = cell_markers.standardize(
        values=sample_fov_markers, field=cell_markers.name, return_mapper=True
    )

    copied_markers = [
        standardized_markers_mapper[m] if m in standardized_markers_mapper else m for m in sample_fov_markers.copy()
    ]

    inspected_markers2 = cell_markers.inspect(values=copied_markers, field=cell_markers.name)
    manually_added_markers = [bt.CellMarker(name=n) for n in inspected_markers2.non_validated]
    validated_markers = bt.CellMarker.from_values(values=inspected_markers2.validated, field="name")

    ln.save(validated_markers)
    ln.save(manually_added_markers)


def create_exploded_ulabels(clinical_data: pd.DataFrame) -> None:
    """Create ULabels for exploded categorical columns.

    Args:
        clinical_data: Clinical data DataFrame
    """
    logger.info("Creating ULabels for exploded columns")
    cols_to_explode = [
        "Ploidy value",
        "17q gain",
        "7q gain",
        "1p LOH",
        "11q LOH",
        "ALK",
        "Other mutations (source)",
        "Genomic studies done",
    ]

    for col in cols_to_explode:
        col_unique_values = set(
            filter(lambda x: len(x) > 0, mit.collapse(clinical_data[col].cat.categories.str.split("|")))
        )
        for v in col_unique_values:
            ln.ULabel(name=v).save()


@app.command(no_args_is_help=True)
def clean_clinical_data(
    clinical_data_path: Annotated[Path, typer.Argument(help="Path to clinical data Excel file")],
    fov_dir: Annotated[Path, typer.Argument(help="Directory containing FOV images")],
    label_dir: Annotated[Path, typer.Argument(help="Directory containing segmentation labels")],
):
    """Clean and validate clinical data for LaminDB.

    Args:
        clinical_data_path
            Path to clinical data Excel file
        fov_dir
            Directory containing FOV images
        label_dir
            Directory containing segmentation labels
    """
    from rich.console import Console

    console = Console()
    ln.track(project="Neuroblastoma")

    console.print(f"Cleaning clinical data from {clinical_data_path}...")
    logger.info("Starting clinical data cleaning")

    # Convert paths to UPath
    clinical_data_path = UPath(clinical_data_path)
    fov_dir = UPath(fov_dir)
    label_dir = UPath(label_dir)

    # Load and clean data
    clinical_data = load_data(clinical_data_path)
    clinical_data = standardize_fovs(clinical_data, fov_dir)
    clinical_data = clean_column_names(clinical_data)
    clinical_data = clean_paired_sequence(clinical_data)
    clinical_data = clean_data_values(clinical_data)

    # Create and validate schema
    schema = create_sample_schema()
    curator = ln.curators.DataFrameCurator(clinical_data, schema)
    curator.validate()

    # Add non-validated categories
    for c in curator.cat.non_validated:
        curator.cat.add_new_from(c)

    curator.validate()
    curator.save_artifact(key="clinical_data.parquet", description="Sample Level Clinical Data")

    # Create ULabels for exploded columns
    create_exploded_ulabels(clinical_data)

    # Validate markers
    validate_markers(fov_dir)

    # Save tissues and ethnicities
    tissues = bt.Tissue.from_values(clinical_data["Tissue"].unique().tolist())
    ethnicities = bt.Ethnicity.from_values(clinical_data["Ethnicity"])
    ln.save(tissues)
    ln.save(ethnicities)

    ln.finish()
    logger.info("Clinical data cleaning complete")


if __name__ == "__main__":
    app()
