from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from models.model import model


def get_geval_metric():
    return GEval(
        name="Correctness",
        criteria="Determine whether the generated answer correctly answers the user's question.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        model=model
    )