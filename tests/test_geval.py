from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from models.openrouter_model import OpenRouterModel

model = OpenRouterModel()


def test_geval():

    metric = GEval(
        name="Correctness",
        criteria="Determine whether the answer correctly answers the user's question.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        model=model
    )

    test_case = LLMTestCase(
        input="What is AWS?",
        actual_output="AWS is Amazon Web Services."
    )

    assert_test(test_case, [metric])