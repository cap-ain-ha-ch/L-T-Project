
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import MercariPreprocessor, FEATURE_COLUMNS


class MercariPricePredictor:
    """Predict listing prices from raw Mercari listing data."""

    def __init__(self, artifact_dir=None):
        if artifact_dir is None:
            artifact_dir = Path(__file__).resolve().parents[1] / "data"
        else:
            artifact_dir = Path(artifact_dir)

        self.preprocessor = joblib.load(
            artifact_dir / "preprocessor.joblib"
        )

        self.text_pipeline = joblib.load(
            artifact_dir / "text_pipeline.joblib"
        )

        saved_model = joblib.load(
            artifact_dir / "models" / "tuned_lightgbm.joblib"
        )
        self.model = saved_model["model"]

        self.feature_names = (
            list(FEATURE_COLUMNS)
            + [f"name_svd_{i}" for i in range(100)]
            + [f"description_svd_{i}" for i in range(100)]
        )

    def prepare_features(self, listings):
        """Convert raw listings into the model's 212 input features."""

        if isinstance(listings, dict):
            listings = pd.DataFrame([listings])
        elif not isinstance(listings, pd.DataFrame):
            listings = pd.DataFrame(listings)

        required = [
            "name",
            "item_condition_id",
            "category_name",
            "brand_name",
            "price",
            "shipping",
            "item_description",
        ]

        # Price is not used as an input feature.
        required.remove("price")

        missing = [col for col in required if col not in listings.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Reuse the fitted preprocessing steps.
        processed = self.preprocessor.transform(listings.copy())

        # Generate text representations using fitted transformers.
        name_vectorizer = self.text_pipeline["name_vectorizer"]
        description_vectorizer = self.text_pipeline[
            "description_vectorizer"
        ]
        name_svd = self.text_pipeline["name_svd"]
        description_svd = self.text_pipeline["description_svd"]

        name_matrix = name_svd.transform(
            name_vectorizer.transform(processed["name_cleaned"])
        )

        description_matrix = description_svd.transform(
            description_vectorizer.transform(
                processed["item_description_cleaned"]
            )
        )

        # Keep the 12 structured features first.
        structured = processed[FEATURE_COLUMNS].reset_index(drop=True)

        # Create the 200 SVD features.
        name_features = pd.DataFrame(
            name_matrix,
            columns=[f"name_svd_{i}" for i in range(100)],
        )

        description_features = pd.DataFrame(
            description_matrix,
            columns=[
                f"description_svd_{i}" for i in range(100)
            ],
        )

        # Combine all 212 features in training order.
        X = pd.concat(
            [structured, name_features, description_features],
            axis=1,
        )

        return X[self.feature_names]

    def predict(self, listings):
        """Return predicted prices in the original price scale."""

        X = self.prepare_features(listings)

        predicted_log_prices = self.model.predict(X)

        # Invert the log1p transformation used during training.
        predicted_prices = np.expm1(predicted_log_prices)

        return np.maximum(predicted_prices, 0)
