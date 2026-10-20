# Mercari Price Prediction

## Overview

A machine learning project that predicts the selling price of second-hand products using listing information such as product name, category, brand, condition, shipping, and description.

The project uses the Mercari dataset and compares multiple regression models to identify an effective price prediction model.

## Features

- Data cleaning and exploratory data analysis
- Missing-value handling and feature engineering
- Text feature extraction using TF-IDF
- Dimensionality reduction using Truncated SVD
- Regression model training and evaluation
- Hyperparameter tuning with RandomizedSearchCV
- Reusable prediction pipeline for individual and multiple listings

## Models

The project evaluates:
- Dummy Regressor
- Ridge Regression
- Random Forest Regressor
- LightGBM Regressor
- Tuned LightGBM Regressor
- Voting Regressor

The tuned LightGBM model is used for final predictions.

## Best Model Performance

| Metric | Validation Score |
|---|---:|
| MAE (log-price space) | 0.4109 |
| RMSE (log-price space) | 0.5411 |

**Note:** These metrics are calculated in log-price space using the log-transformed target. Predicted values are converted back to the original price scale using `expm1`.

## Project Structure

```text
L-T-Project/
├── data/
│   ├── models/
│   ├── train.tsv
│   ├── train_processed.csv
│   ├── valid_processed.csv
│   ├── preprocessor.joblib
│   ├── text_pipeline.joblib
│   └── model_features.joblib
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing_Feature_Engineering.ipynb
│   └── 03_Modeling_Tuning.ipynb
├── src/
│   ├── preprocessing.py
│   └── prediction_pipeline.py
├── .gitignore
└── README.md
```

## Requirements

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- SciPy
- joblib
- Jupyter Notebook

Install the dependencies with:

```bash
pip install pandas numpy scikit-learn lightgbm scipy joblib jupyter
```

## Making Predictions

The prediction pipeline loads the saved preprocessing objects and trained model.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from prediction_pipeline import MercariPricePredictor

predictor = MercariPricePredictor()

listing = {
    "name": "Nike running shoes",
    "item_condition_id": 2,
    "category_name": "Women/Shoes/Athletic",
    "brand_name": "Nike",
    "item_description": "Lightly used running shoes in good condition",
    "shipping": 0,
}

predicted_price = predictor.predict(listing)[0]
print(f"Predicted price: ${predicted_price:.2f}")
```

Run this example from the repository root. The saved model artifacts must be present under `data/`.

## Dataset

The project uses the Mercari Price Suggestion dataset. The raw dataset and generated model artifacts may be too large to store in Git; obtain them separately when reproducing the project.

## Limitations

- Predictions are estimates and may not match actual selling prices.
- Model performance depends on the quality and representativeness of the dataset.
- Validation metrics reported here are in log-price space, not the original currency scale.
