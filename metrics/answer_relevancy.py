from deepeval.metrics import AnswerRelevancyMetric
from models.model import model


def get_answer_relevancy_metric():
    return AnswerRelevancyMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )