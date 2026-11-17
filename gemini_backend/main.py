
# gemini_backend/main.py

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai

# Locate the main PresentationMaker project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the existing project-level .env file
load_dotenv(BASE_DIR / ".env")

# Read Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite"
)

# Create the FastAPI application
app = FastAPI(
    title="PresentationMaker Gemini Backend",
    description="Backend service for AI-powered slide content generation",
    version="1.0.0"
)


# Request format
class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)


# Health-check endpoint
@app.get("/health")
def health_check():

    return {
        "status": "online",
        "gemini_configured": bool(GEMINI_API_KEY)
    }


# Gemini generation endpoint
@app.post("/generate")
def generate_content(request: GenerateRequest):

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured on the server."
        )

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=request.prompt
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response."
            )

        return {
            "success": True,
            "generated_content": response.text
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Gemini generation error: {error}")

        raise HTTPException(
            status_code=502,
            detail="Gemini generation failed. Check server logs."
        )
