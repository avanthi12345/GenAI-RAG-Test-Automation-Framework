from api.api_client import APIClient
from datasets.dataset_loader import DatasetLoader
from metrics.evaluator import Evaluator
from models.openrouter_model import OpenRouterModel

from metrics.answer_relevancy import get_answer_relevancy_metric
from metrics.faithfulness import get_faithfulness_metric
from metrics.hallucination import get_hallucination_metric
from metrics.bias import get_bias_metric
from metrics.toxicity import get_toxicity_metric
from metrics.geval import get_geval_metric


def api_health_check():
    """
    Check whether the API is reachable.
    """

    client = APIClient()

    response = client.get("/")

    print("=" * 60)
    print("API Health Check")
    print("=" * 60)
    print(f"Status Code : {response.status_code}")
    print(f"Response    : {response.text}")


def run_deepeval():
    """
    Execute all supported DeepEval metrics.
    """

    dataset = DatasetLoader(
        "datasets/goldens.json"
    ).load()

    model = OpenRouterModel()

    metrics = [
        get_answer_relevancy_metric(),
        get_faithfulness_metric(),
        get_hallucination_metric(),
        get_bias_metric(),
        get_toxicity_metric(),
        get_geval_metric()
    ]

    evaluator = Evaluator(dataset, model)
    evaluator.evaluate(metrics)


if __name__ == "__main__":

    # Step 1 - Verify API
    api_health_check()

    # Step 2 - Run DeepEval
    run_deepeval()