from models.model import OpenRouterModel

print("Imported OpenRouterModel from:", OpenRouterModel.__module__)

model = OpenRouterModel()


from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from models.model import OpenRouterModel

model = OpenRouterModel()


def test_answer_relevancy():

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=model
    )

    test_case = LLMTestCase(
        input="What is AWS?",
        actual_output="AWS stands for Amazon Web Services."
    )

    assert_test(test_case, [metric])