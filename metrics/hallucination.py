from deepeval.metrics import HallucinationMetric
from models.model import model


def get_hallucination_metric():
    return HallucinationMetric(
        threshold=0.5,
        model=model,
        include_reason=True
    )