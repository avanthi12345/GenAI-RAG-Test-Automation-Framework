from utils.dataset_loader import DatasetLoader

dataset = DatasetLoader.load_dataset(
    "datasets/single_turn/goldens.json"
)

print(dataset)