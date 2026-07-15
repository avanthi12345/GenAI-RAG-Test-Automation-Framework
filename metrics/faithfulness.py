from deepeval.metrics import FaithfulnessMetric
from models.model import model


def get_faithfulness_metric():
    return FaithfulnessMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )