from deepeval.metrics import ToxicityMetric
from models.model import model


def get_toxicity_metric():
    return ToxicityMetric(
        threshold=0.5,
        model=model,
        include_reason=True
    )