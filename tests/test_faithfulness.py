from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from metrics.faithfulness import get_faithfulness_metric


def test_faithfulness():

    metric = get_faithfulness_metric()

    test_case = LLMTestCase(
        input="What is AWS?",
        actual_output="AWS is Amazon Web Services.",
        retrieval_context=[
            "AWS is Amazon Web Services, a cloud computing platform."
        ]
    )

    assert_test(test_case, [metric])