from models.openrouter_model import OpenRouterModel
from deepeval.models.base_model import DeepEvalBaseLLM

model = OpenRouterModel()

print(type(model))
print(isinstance(model, DeepEvalBaseLLM))
print(OpenRouterModel.__mro__)