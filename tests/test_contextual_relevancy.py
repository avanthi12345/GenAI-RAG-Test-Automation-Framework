from deepeval import assert_test
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from models.openrouter_model import OpenRouterModel

model = OpenRouterModel()


def test_contextual_relevancy():

    metric = ContextualRelevancyMetric(
        threshold=0.7,
        model=model
    )

    test_case = LLMTestCase(
        input="What is AWS?",
        actual_output="AWS is Amazon Web Services.",
        retrieval_context=[
            "AWS is Amazon Web Services."
        ]
    )

    assert_test(test_case, [metric])