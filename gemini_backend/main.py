
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BACKEND_ACCESS_TOKEN = os.getenv("BACKEND_ACCESS_TOKEN")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

app = FastAPI(
    title="PresentationMaker Backend",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)


def verify_access_token(authorization: str | None):
    if not BACKEND_ACCESS_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Backend access token is not configured."
        )

    expected = f"Bearer {BACKEND_ACCESS_TOKEN}"

    if not authorization or not secrets.compare_digest(
        authorization, expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing access token."
        )


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "gemini_configured": bool(GEMINI_API_KEY)
    }


@app.post("/generate")
def generate_content(
    request: GenerateRequest,
    authorization: str | None = Header(default=None)
):
    verify_access_token(authorization)

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured."
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
