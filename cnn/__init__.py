"""CNN transit detection experiments package for Assignment 2 Task F."""

from cnn.cnn_classifier import (
    CNNExperimentConfig,
    CNNModelConfig,
    CNNTransitDetector,
    DataConfig,
    EvaluationMetrics,
    FocalLossWithLogits,
    ThresholdMetrics,
    TrainingConfig,
    TransitCNN1D,
)

__all__ = [
    "CNNExperimentConfig",
    "CNNModelConfig",
    "CNNTransitDetector",
    "DataConfig",
    "EvaluationMetrics",
    "FocalLossWithLogits",
    "ThresholdMetrics",
    "TrainingConfig",
    "TransitCNN1D",
]
