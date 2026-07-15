from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

print("API_KEY :", os.getenv("API_KEY"))
print("BASE_URL:", os.getenv("BASE_URL"))
print("MODEL   :", os.getenv("MODEL"))