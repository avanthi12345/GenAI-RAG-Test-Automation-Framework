from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from models.model import OpenRouterModel

model = OpenRouterModel()


def test_hallucination():

    metric = HallucinationMetric(
        threshold=0.5,
        model=model
    )

    test_case = LLMTestCase(
        input="What is AWS?",
        actual_output="AWS is owned by Microsoft.",
        context=[
            "AWS is owned by Amazon."
        ]
    )

    assert_test(test_case, [metric])