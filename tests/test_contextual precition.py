from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

from models.model import OpenRouterModel

model = OpenRouterModel()


def test_contextual_precision():

    metric = ContextualPrecisionMetric(
        threshold=0.7,
        model=model
    )

    test_case = LLMTestCase(
        input="What is AWS?",
        actual_output="AWS is Amazon Web Services.",
        expected_output="AWS is Amazon Web Services.",
        retrieval_context=[
            "AWS is Amazon Web Services."
        ]
    )

    assert_test(test_case, [metric])