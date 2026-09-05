"""
Model layer: training pipeline and inference.

Import-free package init to avoid pulling sklearn/joblib at package import time.
Use:
    from models.predictor import PricePredictor, get_predictor
    from models.train_model import train
"""

__all__ = ["train_model", "predictor"]
