import os

from dotenv import load_dotenv
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

load_dotenv()


class OpenRouterModel(DeepEvalBaseLLM):

    def __init__(self):
        self.model_name = os.getenv("MODEL")

        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name