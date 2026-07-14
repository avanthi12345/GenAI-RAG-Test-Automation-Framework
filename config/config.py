import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    API_KEY = os.getenv("API_KEY")
    MODEL = os.getenv("MODEL")
    BASE_URL = os.getenv("BASE_URL")
    ENV = os.getenv("ENV")
    LOG_LEVEL = os.getenv("LOG_LEVEL")