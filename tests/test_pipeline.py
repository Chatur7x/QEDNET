"""Tests for preprocessing pipeline and training runner."""

import pytest
import numpy as np
from numpy.typing import NDArray

from qednet.preprocess.pipeline import PreprocessingPipeline, create_preprocessing_pipeline
from qednet.data.loader import DatasetLoader, load_dataset, Dataset


class TestPreprocessingPipeline:
    def test_default_pipeline(self, sample_data):
        """Test default preprocessing pipeline."""
        X, y = sample_data
        pipeline = PreprocessingPipeline()
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        assert X_transformed.shape[0] == X.shape[0]
        # PCA reduces to n_components
        assert X_transformed.shape[1] == 8

    def test_no_compression(self, sample_data):
        """Test pipeline with no compression."""
        X, y = sample_data
        pipeline = PreprocessingPipeline(compression="none")
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        assert X_transformed.shape == X.shape

    def test_custom_n_components(self, sample_data):
        """Test pipeline with custom n_components."""
        X, y = sample_data
        pipeline = PreprocessingPipeline(compression="pca", n_components=4)
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        assert X_transformed.shape[1] == 4

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X, y = sample_data
        pipeline = PreprocessingPipeline()
        X_transformed = pipeline.fit_transform(X, y)
        assert X_transformed.shape[0] == X.shape[0]

    def test_not_fitted_error(self, sample_data):
        """Test error when transforming without fitting."""
        X, _ = sample_data
        pipeline = PreprocessingPipeline()
        with pytest.raises(RuntimeError):
            pipeline.transform(X)

    def test_feature_names_out(self, sample_data):
        """Test get_feature_names_out."""
        X, y = sample_data
        pipeline = PreprocessingPipeline()
        pipeline.fit(X, y)
        names = pipeline.get_feature_names_out([f"f{i}" for i in range(X.shape[1])])
        assert len(names) == 8
        assert all(n.startswith("component_") for n in names)


class TestCreatePreprocessingPipeline:
    def test_from_config(self):
        """Test creating pipeline from config dict."""
        config = {
            "scaler": "robust",
            "imputer": "median",
            "compression": "pca",
            "target_dim": 4,
        }
        pipeline = create_preprocessing_pipeline(config)
        assert isinstance(pipeline, PreprocessingPipeline)
        assert pipeline.scaler_type == "robust"
        assert pipeline.imputer_type == "median"
        assert pipeline.compression_type == "pca"
        assert pipeline.n_components == 4


class TestDataLoader:
    def test_load_breast_cancer(self):
        """Test loading breast cancer dataset."""
        dataset = DatasetLoader.load_breast_cancer()
        assert isinstance(dataset, Dataset)
        assert dataset.name == "breast_cancer"
        assert dataset.X.shape[1] == 30  # 30 features
        assert len(dataset.feature_names) == 30
        assert len(dataset.target_names) == 2

    def test_load_unknown_dataset_raises(self):
        """Test error for unknown dataset."""
        with pytest.raises(ValueError):
            DatasetLoader.load_dataset("unknown")

    def test_split_dataset(self):
        """Test dataset splitting."""
        dataset = DatasetLoader.load_breast_cancer()
        train_ds, test_ds = DatasetLoader.split(dataset, test_size=0.2, random_state=42)
        assert isinstance(train_ds, Dataset)
        assert isinstance(test_ds, Dataset)
        assert len(train_ds.X) + len(test_ds.X) == len(dataset.X)
        assert train_ds.feature_names == dataset.feature_names


class TestLoadDataset:
    def test_load_with_config(self):
        """Test load_dataset function with config."""
        config = {"name": "breast_cancer", "test_size": 0.2, "random_state": 42}
        train_ds, test_ds = load_dataset(config)
        assert isinstance(train_ds, Dataset)
        assert isinstance(test_ds, Dataset)