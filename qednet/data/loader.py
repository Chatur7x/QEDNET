"""Data loading utilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    X: NDArray
    y: NDArray
    feature_names: list[str]
    target_names: list[str]
    name: str = "unknown"


class DatasetLoader:
    """Load and prepare datasets for QED-Net."""

    @staticmethod
    def load_breast_cancer() -> Dataset:
        """Load the Wisconsin Breast Cancer dataset."""
        data = load_breast_cancer()
        return Dataset(
            X=data.data.astype(np.float32),
            y=data.target.astype(np.int64),
            feature_names=list(data.feature_names),
            target_names=list(data.target_names),
            name="breast_cancer",
        )

    @staticmethod
    def load_dataset(name: str) -> Dataset:
        """Load a dataset by name."""
        if name == "breast_cancer":
            return DatasetLoader.load_breast_cancer()
        raise ValueError(f"Unknown dataset: {name}")

    @staticmethod
    def split(
        dataset: Dataset,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> tuple[Dataset, Dataset]:
        """Split dataset into train and test sets."""
        stratify_y = dataset.y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )
        train_ds = Dataset(
            X_train, y_train, dataset.feature_names, dataset.target_names, f"{dataset.name}_train"
        )
        test_ds = Dataset(
            X_test, y_test, dataset.feature_names, dataset.target_names, f"{dataset.name}_test"
        )
        return train_ds, test_ds


def load_dataset(config: dict) -> tuple[Dataset, Dataset]:
    """Load and split dataset based on config."""
    dataset_name = config.get("name", "breast_cancer")
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)

    dataset = DatasetLoader.load_dataset(dataset_name)
    train_ds, test_ds = DatasetLoader.split(dataset, test_size, random_state)
    return train_ds, test_ds
