from datasets.dataset_loader import DatasetLoader
from metrics.answer_relevancy import get_answer_relevancy_metric
from metrics.evaluator import Evaluator
from models.openrouter_model import OpenRouterModel


def test_answer_relevancy():

    dataset = DatasetLoader(
        "datasets/single_turn/goldens.json"
    ).load()

    model = OpenRouterModel()

    metric = get_answer_relevancy_metric()

    evaluator = Evaluator(
        dataset=dataset,
        model=model
    )

    evaluator.evaluate([metric])