import json

from deepeval.dataset import EvaluationDataset
from deepeval.dataset.golden import Golden


class DatasetLoader:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> EvaluationDataset:

        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        goldens = []

        for item in data:

            golden = Golden(
                input=item["input"],
                expected_output=item["expected_output"],
                context=item.get("context", []),
                retrieval_context=item.get("retrieval_context", [])
            )

            goldens.append(golden)

        return EvaluationDataset(goldens=goldens)