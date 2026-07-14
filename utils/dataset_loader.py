import json


class DatasetLoader:

    @staticmethod
    def load_dataset(file_path):
        """
        Load Golden Dataset JSON
        """

        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)