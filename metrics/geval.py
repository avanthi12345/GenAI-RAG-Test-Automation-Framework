from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from datasets.dataset_loader import DatasetLoader
from metrics.evaluator import Evaluator
from models.openrouter_model import OpenRouterModel


def get_geval_metric():
    """
    Returns the configured GEval metric.
    """
    model = OpenRouterModel()

    return GEval(
        name="Overall Quality",
        criteria="""
        Evaluate the response based on:
        - Correctness
        - Relevance
        - Completeness
        - Clarity
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        model=model
    )


def run():
    """
    Runs GEval evaluation.
    """

    dataset = DatasetLoader(
        "datasets/single_turn/goldens.json"
    ).load()

    model = OpenRouterModel()

    metric = get_geval_metric()

    evaluator = Evaluator(
        dataset=dataset,
        model=model
    )

    evaluator.evaluate([metric])


if __name__ == "__main__":
    run()