"""CNN transit detection experiments package for Assignment 2 Task F."""

from cnn.cnn_classifier import (
    CNN,
    DEFAULT_CONFIG,
    MODEL_KEYS,
    CNNClassifier,
    CNNDetection,
    CNNTransitDetector,
    FocalLossWithLogits,
    TransitCNN1D,
)

__all__ = [
    "CNN",
    "CNNClassifier",
    "CNNDetection",
    "CNNTransitDetector",
    "DEFAULT_CONFIG",
    "FocalLossWithLogits",
    "MODEL_KEYS",
    "TransitCNN1D",
]
