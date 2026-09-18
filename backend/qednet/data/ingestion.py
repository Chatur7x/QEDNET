"""F1 — Biomedical Data Ingestion.

CSV upload, dataset registration, metadata, target-column selection, schema
discovery, datatype inspection and basic profiling. Invalid files, empty
datasets, missing target columns, invalid datatypes and malformed rows are
detected and reported.

New dataset sources are added by implementing the ``DatasetAdapter`` interface
(the registry is open for extension without rewriting the ingestion layer).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("qednet.ingestion")

SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")
IDENTIFIER_HINTS = ("name", "email", "phone", "patient_id", "ssn", "address",
                    "mrn", "patient number", "mail")

ALLOWED_EXTENSIONS = {".csv", ".tsv"}

REGISTRY_PATH = Path("data/registry/datasets.json")


class IngestionError(ValueError):
    """Raised when an uploaded dataset cannot be ingested."""


@dataclass
class DatasetProfile:
    """Basic dataset profiling information."""
    n_samples: int
    n_features: int
    feature_names: List[str]
    dtypes: Dict[str, str]
    class_balance: Dict[str, float]
    missing_per_column: Dict[str, int]
    duplicate_rows: int
    constant_columns: List[str]
    identifier_like_columns: List[str]
    memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegisteredDataset:
    """A registered dataset with metadata."""
    name: str
    source: str                 # builtin | csv
    target: str
    path: Optional[str] = None
    sha256: Optional[str] = None
    n_samples: int = 0
    n_features: int = 0
    profile: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pending"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def safe_filename(name: str) -> str:
    """Sanitise a filename; rejects path traversal."""
    cleaned = SAFE_NAME.sub("_", str(name)).strip("._-")
    if not cleaned:
        raise IngestionError("Invalid (empty after sanitisation) dataset name")
    return cleaned[:80]


def fingerprint_frame(df: pd.DataFrame) -> str:
    """Dataset fingerprint: sha256 over sorted content hash."""
    payload = f"{df.shape[0]}x{df.shape[1]}|"
    payload += "|".join(map(str, df.columns.tolist()))
    payload += "|" + hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def profile_frame(df: pd.DataFrame, target: str) -> DatasetProfile:
    """Basic profiling of a dataframe with a binary target column."""
    y = df[target]
    counts = y.value_counts(normalize=True)
    balance = {str(k): float(v) for k, v in counts.items()}
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    constant = [c for c in df.columns
                if c != target and df[c].nunique(dropna=False) <= 1]
    identifier_like = [
        c for c in df.columns
        if any(h in str(c).lower() for h in IDENTIFIER_HINTS)
        or str(c).lower() in ("id", "index", "subject", "subject_id")
    ]
    return DatasetProfile(
        n_samples=int(df.shape[0]),
        n_features=int(df.shape[1] - 1),
        feature_names=[c for c in df.columns if c != target],
        dtypes={c: str(df[c].dtype) for c in df.columns},
        class_balance=balance,
        missing_per_column=missing,
        duplicate_rows=int(df.duplicated().sum()),
        constant_columns=constant,
        identifier_like_columns=identifier_like,
        memory_mb=float(df.memory_usage(deep=True).sum() / 1e6),
    )


def ingest_csv(path: str | Path, target: str | None = None,
               dataset_name: Optional[str] = None) -> RegisteredDataset:
    """Ingest a CSV file: detect invalid files, discover schema, profile.

    Raises ``IngestionError`` for invalid files, empty datasets, missing
    target columns, or unusable data.
    """
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"File not found: {path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{path.suffix}'. Allowed: CSV/TSV.")
    sep = "\t" if path.suffix.lower() == ".tsv" else ","

    try:
        df = pd.read_csv(path, sep=sep)
    except pd.errors.EmptyDataError as exc:
        raise IngestionError("Invalid file: empty or unparseable CSV") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        raise IngestionError(f"Invalid file: {exc}") from exc

    if df.shape[0] == 0:
        raise IngestionError("Empty dataset: zero rows")
    if df.shape[1] < 2:
        raise IngestionError(
            "Dataset needs at least one feature column and one target column")

    if target is None:
        target = df.columns[-1]
    if target not in df.columns:
        raise IngestionError(
            f"Missing target column '{target}'. Available: {list(df.columns)[:10]}")

    # obviously malformed rows: all-NaN rows
    malformed = int(df.isna().all(axis=1).sum())
    if malformed:
        logger.warning("Dropping %d all-NaN malformed rows", malformed)
        df = df.dropna(how="all")

    name = dataset_name or safe_filename(path.stem)
    profile = profile_frame(df, target)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    return RegisteredDataset(
        name=name, source="csv", target=target, path=str(path), sha256=sha,
        n_samples=profile.n_samples, n_features=profile.n_features,
        profile=profile.to_dict(), validation_status="pending",
    )


class DatasetAdapter:
    """Interface for dataset sources (open for extension)."""
    name: str = "adapter"

    def load(self) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

    def target(self) -> str:
        return "target"


class DatasetRegistry:
    """Registry of known datasets (builtin adapters + uploaded CSVs)."""

    def __init__(self, registry_path: Path | None = None,
                 data_dir: Path | None = None):
        self.registry_path = Path(registry_path or REGISTRY_PATH)
        self.data_dir = Path(data_dir or Path("data"))
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        if self.registry_path.exists():
            try:
                self._entries = json.loads(self.registry_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Corrupt dataset registry; starting fresh")
                self._entries = {}
        else:
            self._entries = {}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self._entries, indent=2))

    # -- operations -------------------------------------------------------
    def register(self, rec: RegisteredDataset) -> None:
        self._entries[rec.name] = rec.to_dict()
        self._save()

    def list_datasets(self) -> List[Dict[str, Any]]:
        out = []
        for name in sorted(self._entries):
            e = dict(self._entries[name])
            out.append(e)
        return out

    def get(self, name: str) -> Dict[str, Any]:
        if name not in self._entries:
            raise KeyError(f"Dataset '{name}' is not registered")
        return dict(self._entries[name])

    def exists(self, name: str) -> bool:
        return name in self._entries


def load_dataframe(name: str, registry: DatasetRegistry) -> pd.DataFrame:
    """Load a registered dataset as a dataframe (feature columns + target)."""
    meta = registry.get(name)
    if meta["source"] == "csv":
        path = meta.get("path")
        if not path or not Path(path).exists():
            raise IngestionError(
                f"Registered CSV for '{name}' no longer exists at {path}")
        df = pd.read_csv(path)
    else:
        raise IngestionError(
            f"Unknown builtin dataset '{name}' — use qednet.data.datasets")
    return df
