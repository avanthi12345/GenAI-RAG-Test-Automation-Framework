from deepeval.metrics import ContextualRecallMetric

from datasets.dataset_loader import DatasetLoader
from metrics.evaluator import Evaluator
from models.openrouter_model import OpenRouterModel


def get_contextual_recall_metric():
    """
    Returns the configured Contextual Recall metric.
    """
    model = OpenRouterModel()

    return ContextualRecallMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )


def run():
    """
    Runs Contextual Recall evaluation.
    """

    dataset = DatasetLoader(
        "datasets/single_turn/goldens.json"
    ).load()

    model = OpenRouterModel()

    metric = get_contextual_recall_metric()

    evaluator = Evaluator(
        dataset=dataset,
        model=model
    )

    evaluator.evaluate([metric])


if __name__ == "__main__":
    run()