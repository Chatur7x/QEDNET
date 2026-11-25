"""Data ingestion and loading utilities."""

from qednet.data.deidentification import (
    generalize_age,
    hash_identifiers,
    k_anonymize,
    l_diversity,
    remove_identifiers,
    suppress_rare_categories,
)
from qednet.data.ingestion import DatasetBundle, DatasetRegistry
from qednet.data.loader import DatasetLoader, load_dataset
from qednet.data.validation import DATASET_METADATA, DATASET_SCHEMAS

__all__ = [
    "load_dataset",
    "DatasetLoader",
    "DatasetBundle",
    "DatasetRegistry",
    "DATASET_METADATA",
    "DATASET_SCHEMAS",
    "hash_identifiers",
    "remove_identifiers",
    "generalize_age",
    "suppress_rare_categories",
    "k_anonymize",
    "l_diversity",
]
