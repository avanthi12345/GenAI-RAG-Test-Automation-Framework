from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from metrics.answer_relevancy import get_answer_relevancy_metric


def test_answer_relevancy():

    metric = get_answer_relevancy_metric()

    test_case = LLMTestCase(
        input="What is AWS?",
        actual_output="AWS stands for Amazon Web Services."
    )

    assert_test(test_case, [metric])