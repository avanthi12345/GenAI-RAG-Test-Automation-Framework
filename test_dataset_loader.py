from datasets.dataset_loader import DatasetLoader

dataset = DatasetLoader(
    "datasets/single_turn/goldens.json"
).load()

print(f"Total Goldens : {len(dataset.goldens)}")

for golden in dataset.goldens:
    print("-" * 50)
    print("Input :", golden.input)
    print("Expected :", golden.expected_output)