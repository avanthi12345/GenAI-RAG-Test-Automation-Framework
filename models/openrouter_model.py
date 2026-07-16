import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class OpenRouterModel(DeepEvalBaseLLM):
    """
    Custom OpenRouter model compatible with DeepEval.
    """

    def __init__(self):
        self.model_name = os.getenv("MODEL")

        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

    def load_model(self):
        """
        Return the initialized OpenAI/OpenRouter client.
        """
        return self.client

    def generate(self, prompt: str, schema=None):
        """
        Generate model response.

        If DeepEval requests structured output (schema),
        use the parse API. Otherwise return plain text.
        """

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Structured output (GEval, etc.)
        if schema is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=schema
            )

            return response.choices[0].message.parsed

        # Normal completion
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None):
        """
        Async wrapper required by DeepEval.
        """
        return self.generate(prompt, schema)

    def get_model_name(self):
        """
        Return the model name.
        """
        return self.model_name