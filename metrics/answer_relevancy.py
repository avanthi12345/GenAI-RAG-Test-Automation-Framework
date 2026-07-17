from deepeval.metrics import AnswerRelevancyMetric

from datasets.dataset_loader import DatasetLoader
from metrics.evaluator import Evaluator
from models.openrouter_model import OpenRouterModel


def get_answer_relevancy_metric():
    """
    Returns the configured Answer Relevancy metric.
    """
    model = OpenRouterModel()

    return AnswerRelevancyMetric(
        threshold=0.7,
        model=model,
        include_reason=True
    )


def run():
    """
    Runs Answer Relevancy evaluation using the dataset.
    """

    # Choose the dataset to evaluate
    # dataset = DatasetLoader.load_positive_goldens()
    # dataset = DatasetLoader.load_negative_goldens()
    # dataset = DatasetLoader.load_edge_goldens()
    dataset = DatasetLoader.load_all_goldens()

    model = OpenRouterModel()

    metric = get_answer_relevancy_metric()

    evaluator = Evaluator(
        dataset=dataset,
        model=model
    )

    evaluator.evaluate([metric])


if __name__ == "__main__":
    run()