from datasets.dataset_loader import DatasetLoader
from metrics.contextual_recall import get_contextual_recall_metric
from metrics.evaluator import Evaluator
from models.openrouter_model import OpenRouterModel


def test_contextual_recall():

    dataset = DatasetLoader(
        "datasets/goldens.json"
    ).load()

    model = OpenRouterModel()

    metric = get_contextual_recall_metric()

    evaluator = Evaluator(dataset, model)
    evaluator.evaluate([metric])