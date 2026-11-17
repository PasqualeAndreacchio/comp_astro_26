"""Classifier interfaces for the detection course exercises."""

from .random_forest import RandomForestClassifier, RandomForestDetection

__all__ = ["RandomForestClassifier", "RandomForestDetection", "CNNClassifier"]


class CNNClassifier:
    """Placeholder for the CNN exercise."""

    def fit(self, X, y):
        raise NotImplementedError("CNN training is a course exercise")
