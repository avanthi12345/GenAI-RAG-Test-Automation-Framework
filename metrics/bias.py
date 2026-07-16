from deepeval.metrics import BiasMetric

from datasets.dataset_loader import DatasetLoader
from metrics.evaluator import Evaluator
from models.openrouter_model import OpenRouterModel


def get_bias_metric():
    """
    Returns the configured Bias metric.
    """
    model = OpenRouterModel()

    return BiasMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )


def run():
    """
    Runs Bias evaluation.
    """

    dataset = DatasetLoader(
        "datasets/single_turn/goldens.json"
    ).load()

    model = OpenRouterModel()

    metric = get_bias_metric()

    evaluator = Evaluator(
        dataset=dataset,
        model=model
    )

    evaluator.evaluate([metric])


if __name__ == "__main__":
    run()