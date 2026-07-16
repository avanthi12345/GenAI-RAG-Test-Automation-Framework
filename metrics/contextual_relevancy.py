from deepeval.metrics import ContextualRelevancyMetric

from datasets.dataset_loader import DatasetLoader
from metrics.evaluator import Evaluator
from models.openrouter_model import OpenRouterModel


def get_contextual_relevancy_metric():
    """
    Returns the configured Contextual Relevancy metric.
    """
    model = OpenRouterModel()

    return ContextualRelevancyMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )


def run():
    """
    Runs Contextual Relevancy evaluation.
    """

    dataset = DatasetLoader(
        "datasets/single_turn/goldens.json"
    ).load()

    model = OpenRouterModel()

    metric = get_contextual_relevancy_metric()

    evaluator = Evaluator(
        dataset=dataset,
        model=model
    )

    evaluator.evaluate([metric])


if __name__ == "__main__":
    run()