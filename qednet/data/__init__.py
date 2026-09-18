"""Data ingestion and loading utilities."""
from qednet.data.loader import load_dataset, DatasetLoader
from qednet.data.ingestion import DatasetBundle, DatasetRegistry
from qednet.data.validation import DATASET_METADATA, DATASET_SCHEMAS
from qednet.data.deidentification import (
    hash_identifiers,
    remove_identifiers,
    generalize_age,
    suppress_rare_categories,
    k_anonymize,
    l_diversity,
)

__all__ = [
    "load_dataset", "DatasetLoader",
    "DatasetBundle", "DatasetRegistry",
    "DATASET_METADATA", "DATASET_SCHEMAS",
    "hash_identifiers", "remove_identifiers", "generalize_age",
    "suppress_rare_categories", "k_anonymize", "l_diversity",
]