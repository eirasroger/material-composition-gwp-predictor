"""
GHG (Greenhouse Gas) Predictor — entrypoint.
=========================================
Predicts ghg from material composition, product category, circularity-origin,
and end-of-life pathways.

Materials are embedded via Word2Vec-style / fastText-style vectors (300-d)
and weighted by mass fractions. Category is one-hot encoded and concatenated
with the material embedding and circularity features.

What this pipeline does:
- Keeps only products with reference_unit == "kg"
- Filters out categories with fewer than MIN_CATEGORY_COUNT products
- Drops any product with invalid material percentages
- Drops any product with invalid/missing target values
- Keeps only products with GHG_MIN <= ghg <= GHG_MAX
- Uses circularity-origin and end-of-life percentages as extra numeric features
- Applies British->American English normalisation on material tokens
- Filters stopwords from material token sequences
- Uses zero-vector fallback for OOV tokens (instead of silently skipping)
- Assigns equal mass fractions when all percentages are missing or zero
- L2-normalises the final product embedding before concatenation
- Stratifies train/val/test splits by category
- Trains a regression model with log1p target transform
- Prints epoch-by-epoch training/validation metrics
- Evaluates on a held-out test set with per-category breakdown
- Saves plots and diagnostics
- Includes a prediction helper for inference

JSON structure expected:
  product["c_pcr"]          -> standardised category string
  product["product_integrity"]["materials"] -> list of {name, percentage}
  product["ghg_footprint"]["total_ghg"]     -> float target
  product["cyclability"]                    -> dict with lowercase future_use_* keys

Dependencies:
    pip install numpy scikit-learn torch matplotlib

Code layout:
    src/config.py               paths, hyperparameters, constants, seeds
    src/utils.py                small shared helpers (tokenisers, safe_float, r2_safe, ...)
    src/data/loader.py          raw load + reference-unit filter + category index
    src/data/preprocessing.py   per-product validation, circularity features
    src/data/features.py        feature-matrix construction
    src/embeddings/vocab.py     pretrained-vector loaders (w2v / fastText / custom .vec)
    src/embeddings/encode.py    material/product/category encoders
    src/model/dataset.py        torch Dataset wrapper
    src/model/network.py        GHGNet MLP
    src/train/trainer.py        training loop with early stopping
    src/train/evaluator.py      held-out + per-category evaluation
    src/reporting/plots.py      diagnostic plots and JSON dump
    src/inference/predict.py    single-product GHG prediction
    src/pipeline.py             end-to-end orchestration
"""

from src.pipeline import run


if __name__ == "__main__":
    run()
