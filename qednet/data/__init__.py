"""D2 Data Ingestion & Validation public API."""

from .ingestion import DatasetBundle, DatasetRegistry
from .registry import DATASET_METADATA, DATASET_SCHEMAS, deidentify
from .deidentification import (
    hash_identifiers,
    remove_identifiers,
    generalize_age,
    suppress_rare_categories,
    k_anonymize,
    l_diversity,
)

__all__ = [
    "DatasetBundle",
    "DatasetRegistry",
    "deidentify",
    "DATASET_METADATA",
    "DATASET_SCHEMAS",
    "hash_identifiers",
    "remove_identifiers",
    "generalize_age",
    "suppress_rare_categories",
    "k_anonymize",
    "l_diversity",
]