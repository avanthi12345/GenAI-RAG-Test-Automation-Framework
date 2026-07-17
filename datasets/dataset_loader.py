import json
from pathlib import Path

from deepeval.dataset import EvaluationDataset
from deepeval.dataset.golden import Golden


class DatasetLoader:

    BASE_DIR = Path(__file__).resolve().parent

    @staticmethod
    def load_json(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @classmethod
    def _build_dataset(cls, file_path):
        data = cls.load_json(file_path)

        dataset = EvaluationDataset()

        for item in data:
            dataset.add_golden(
                Golden(
                    input=item["input"],
                    expected_output=item["expected_output"],
                    context=item.get("context", []),
                    retrieval_context=item.get("retrieval_context", [])
                )
            )

        return dataset

    @classmethod
    def load_goldens(cls):
        return cls._build_dataset(
            cls.BASE_DIR / "single_turn" / "goldens.json"
        )

    @classmethod
    def load_negative_goldens(cls):
        return cls._build_dataset(
            cls.BASE_DIR / "single_turn" / "negative_goldens.json"
        )

    @classmethod
    def load_edge_goldens(cls):
        return cls._build_dataset(
            cls.BASE_DIR / "single_turn" / "edge_goldens.json"
        )

    @classmethod
    def load_all_goldens(cls):
        dataset = EvaluationDataset()

        for loader in [
            cls.load_goldens,
            cls.load_negative_goldens,
            cls.load_edge_goldens
        ]:
            temp_dataset = loader()

            for golden in temp_dataset.goldens:
                dataset.add_golden(golden)

        return dataset