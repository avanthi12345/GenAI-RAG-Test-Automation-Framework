from deepeval import evaluate
from deepeval.test_case import LLMTestCase


class Evaluator:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def build_test_cases(self):
        """
        Build DeepEval test cases from the EvaluationDataset.
        """

        test_cases = []

        for golden in self.dataset.goldens:

            # Generate actual output from the model
            actual_output = self.model.generate(golden.input)

            test_case = LLMTestCase(
                input=golden.input,
                actual_output=actual_output,
                expected_output=golden.expected_output,
                context=getattr(golden, "context", []),
                retrieval_context=getattr(golden, "retrieval_context", [])
            )

            test_cases.append(test_case)

        return test_cases

    def evaluate(self, metrics):
        """
        Execute DeepEval metrics.
        """

        test_cases = self.build_test_cases()

        results = evaluate(
            test_cases=test_cases,
            metrics=metrics
        )

        return results