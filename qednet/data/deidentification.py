"""De-identification utilities (D2)."""

from __future__ import annotations

import hashlib

import pandas as pd


def hash_identifiers(df: pd.DataFrame, columns: list[str], salt: str = "") -> pd.DataFrame:
    """Replace identifier columns with salted hashes."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .apply(lambda x: hashlib.sha256((salt + x).encode()).hexdigest()[:16])
            )
    return out


def remove_identifiers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop identifier columns entirely."""
    return df.drop(columns=[c for c in columns if c in df.columns], errors="ignore")


def generalize_age(df: pd.DataFrame, column: str = "age", bins: int = 10) -> pd.DataFrame:
    """Bin age into ranges for privacy."""
    out = df.copy()
    if column in out.columns:
        out[column] = pd.cut(out[column], bins=bins)
    return out


def suppress_rare_categories(
    df: pd.DataFrame, column: str, threshold: int = 5, replacement: str = "OTHER"
) -> pd.DataFrame:
    """Replace rare categorical values with a generic label."""
    out = df.copy()
    if column in out.columns:
        counts = out[column].value_counts()
        rare = counts[counts < threshold].index
        out[column] = out[column].replace(rare, replacement)
    return out


def k_anonymize(
    df: pd.DataFrame, quasi_identifiers: list[str], k: int = 5, suppress: bool = True
) -> pd.DataFrame:
    """
    Basic k-anonymity: suppress rows in groups smaller than k.
    For production, use a dedicated library like `pycanon` or `sdc`.
    """
    out = df.copy()
    grouped = out.groupby(quasi_identifiers).size().reset_index(name="count")
    valid_groups = grouped[grouped["count"] >= k][quasi_identifiers]
    out = out.merge(valid_groups, on=quasi_identifiers, how="inner")
    return out


def l_diversity(
    df: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive_column: str,
    l: int = 2,  # noqa: E741
) -> pd.DataFrame:
    """
    Basic l-diversity: ensure each equivalence class has >= l distinct sensitive values.
    """
    out = df.copy()
    grouped = (
        out.groupby(quasi_identifiers)[sensitive_column].nunique().reset_index(name="diversity")
    )
    valid_groups = grouped[grouped["diversity"] >= l][quasi_identifiers]
    out = out.merge(valid_groups, on=quasi_identifiers, how="inner")
    return out
