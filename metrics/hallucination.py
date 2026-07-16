from deepeval.metrics import HallucinationMetric

from datasets.dataset_loader import DatasetLoader
from metrics.evaluator import Evaluator
from models.openrouter_model import OpenRouterModel


def get_hallucination_metric():
    """
    Returns the configured Hallucination metric.
    """
    model = OpenRouterModel()

    return HallucinationMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )


def run():
    """
    Runs Hallucination evaluation.
    """

    dataset = DatasetLoader(
        "datasets/single_turn/goldens.json"
    ).load()

    model = OpenRouterModel()

    metric = get_hallucination_metric()

    evaluator = Evaluator(
        dataset=dataset,
        model=model
    )

    evaluator.evaluate([metric])


if __name__ == "__main__":
    run()