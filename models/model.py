import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

# Load .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


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

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # DeepEval requests structured output
        if schema is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=schema
            )
            return response.choices[0].message.parsed

        # Normal text generation
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


model = OpenRouterModel()