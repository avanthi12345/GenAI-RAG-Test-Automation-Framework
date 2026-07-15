from deepeval.metrics import BiasMetric
from models.model import model


def get_bias_metric():
    return BiasMetric(
        threshold=0.5,
        model=model,
        include_reason=True
    )