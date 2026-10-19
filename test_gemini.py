# OPTIONAL - Quick check that your Gemini API key works.
import os

from dotenv import load_dotenv
from google import genai

from config import ENV_FILE, GEMINI_MODEL

load_dotenv(ENV_FILE)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(f"GEMINI_API_KEY is missing from {ENV_FILE}")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents="Reply with one short sentence confirming you are working.",
)

print(f"Model: {GEMINI_MODEL}")
print(response.text)
