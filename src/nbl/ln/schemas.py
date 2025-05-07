import bionty as bt
import lamindb as ln

clinical_schema = schema = ln.Schema(
    name="Neuroblastoma Clinical Schema",
    features=[
        ln.Feature(name="FOV", dtype=ln.ULabel, description="Identifier for the imaging Field of View").save(),
        ln.Feature(name="patient_ID", dtype="int", description="Unique identifier for the patient").save(),
        ln.Feature(name="UID", dtype=ln.ULabel, description="Unique identifier for the sample or record").save(),
        ln.Feature(
            name="Paired sequence", dtype="bool", description="Indicates whether the sequence is paired or not"
        ).save(),
        ln.Feature(
            name="Age (days) at Diagnosis", dtype="int", description="Patient's age in days at the time of diagnosis"
        ).save(),
        ln.Feature(
            name="Classification",
            dtype=ln.ULabel,
            description="Clinical or histological classification of the specimen/tumor",
        ).save(),
        ln.Feature(
            name="Age (days) at Biopsy",
            dtype="int",
            nullable=True,
            description="Patient's age in days at the time of biopsy",
        ).save(),
        ln.Feature(name="Sex", dtype=ln.ULabel, nullable=True, description="Biological sex of the patient").save(),
        ln.Feature(
            name="Ethnicity",
            dtype=bt.Ethnicity,
            nullable=True,
            description="Self-reported or assigned ethnicity of the patient",
        ).save(),
        ln.Feature(name="Tissue", dtype=bt.Tissue, description="Type of tissue biopsied or imaged").save(),
        ln.Feature(
            name="VMA (g Cr)",
            dtype="float",
            nullable=True,
            description="vanillylmandelic acid (VMA) measurements expressed in relation to grams of creatinine (g Cr) in urine samples.",
        ).save(),
        ln.Feature(
            name="HVA (g Cr)",
            dtype="float",
            nullable=True,
            description="homovanillic acid (HVA) measurements expressed in relation to grams of creatinine (g Cr) in urine samples.",
        ).save(),
        ln.Feature(
            name="HVA/VMA days from biopsy",
            dtype="float",
            nullable=True,
            description="Number of days between HVA/VMA measurement and biopsy",
        ).save(),
        ln.Feature(
            name="Clinical presentation",
            dtype="str",
            nullable=True,
            description="Description of the patient's symptoms and presentation at diagnosis",
        ).save(),
        ln.Feature(
            name="Risk", dtype=ln.ULabel, description="Assigned clinical risk group (e.g., low, intermediate, high)"
        ).save(),
        ln.Feature(
            name="INSS stage",
            dtype=ln.ULabel,
            nullable=True,
            description="International Neuroblastoma Staging System (INSS) stage",
        ).save(),
        ln.Feature(
            name="INRG stage",
            dtype=ln.ULabel,
            nullable=True,
            description="International Neuroblastoma Risk Group (INRG) staging system stage",
        ).save(),
        ln.Feature(
            name="Ploidy value",
            dtype=ln.ULabel,
            nullable=True,
            description="Ploidy status of the tumor cells (e.g., diploid, hyperdiploid)",
        ).save(),
        ln.Feature(
            name="MKI",
            dtype=ln.ULabel,
            nullable=True,
            description="Mitotic-Karyorrhectic Index (MKI) value",
        ).save(),
        ln.Feature(
            name="Degree of differentiation",
            dtype=ln.ULabel,
            nullable=True,
            description="Histological degree of tumor cell differentiation",
        ).save(),
        ln.Feature(
            name="Histolgic classification - INPC",
            dtype=ln.ULabel,
            nullable=True,
            description="International Neuroblastoma Pathology Classification (INPC) category",
        ).save(),
        ln.Feature(
            name="Genomics source",
            dtype=ln.ULabel,
            nullable=True,
            description="Source material used for genomic analysis (e.g., tumor, blood)",
        ).save(),
        ln.Feature(
            name="MYCN amplification",
            dtype="bool",
            nullable=True,
            description="Presence (True) or absence (False) of MYCN gene amplification",
        ).save(),
        ln.Feature(
            name="17q gain", dtype=ln.ULabel, nullable=True, description="Presence or status of chromosome 17q gain"
        ).save(),
        ln.Feature(
            name="7q gain", dtype=ln.ULabel, nullable=True, description="Presence or status of chromosome 7q gain"
        ).save(),
        ln.Feature(
            name="1p LOH",
            dtype=ln.ULabel,
            nullable=True,
            description="Presence or status of Loss of Heterozygosity (LOH) on chromosome 1p",
        ).save(),
        ln.Feature(
            name="11q LOH",
            dtype=ln.ULabel,
            nullable=True,
            description="Presence or status of Loss of Heterozygosity (LOH) on chromosome 11q",
        ).save(),
        ln.Feature(
            name="ALK",
            dtype=ln.ULabel,
            nullable=True,
            description="Status of ALK (Anaplastic Lymphoma Kinase) gene alteration (e.g., mutation, amplification)",
        ).save(),
        ln.Feature(
            name="Other mutations (source)",
            dtype=ln.ULabel,
            nullable=True,
            description="Details of other relevant mutations identified and their source",
        ).save(),
        ln.Feature(
            name="Genomic studies done",
            dtype=ln.ULabel,
            nullable=True,
            description="Description of the types of genomic studies performed",
        ).save(),
        ln.Feature(
            name="treatment btw biopsies",
            dtype="str",
            nullable=True,
            description="Details of any treatment received between biopsies",
        ).save(),
        ln.Feature(
            name="OS time (days)",
            dtype="int",
            nullable=True,
            description="Overall Survival (OS) time in days from diagnosis",
        ).save(),
    ],
).save()
