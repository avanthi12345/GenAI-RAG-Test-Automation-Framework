from deepeval.metrics import FaithfulnessMetric

from datasets.dataset_loader import DatasetLoader
from metrics.evaluator import Evaluator
from models.openrouter_model import OpenRouterModel


def get_faithfulness_metric():
    """
    Returns the configured Faithfulness metric.
    """
    model = OpenRouterModel()

    return FaithfulnessMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )


def run():
    """
    Runs Faithfulness evaluation.
    """

    dataset = DatasetLoader(
        "datasets/single_turn/goldens.json"
    ).load()

    model = OpenRouterModel()

    metric = get_faithfulness_metric()

    evaluator = Evaluator(
        dataset=dataset,
        model=model
    )

    evaluator.evaluate([metric])


if __name__ == "__main__":
    run()