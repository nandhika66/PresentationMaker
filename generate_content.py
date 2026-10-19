# Sends the retrieved slides + user request to Gemini and gets back
# structured slide content as JSON.

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import ENV_FILE, GEMINI_MODEL

load_dotenv(ENV_FILE)

_client = None


def get_client():
    """Create the Gemini client only when it is first needed."""
    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                f"GEMINI_API_KEY is missing. Add it to {ENV_FILE}"
            )
        _client = genai.Client(api_key=api_key)

    return _client


RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "presentation_title": {"type": "STRING"},
        "slides": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "key_points": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "source_slides": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "source_file": {"type": "STRING"},
                                "slide_number": {"type": "STRING"},
                            },
                            "required": ["source_file", "slide_number"],
                        },
                    },
                },
                "required": ["title", "key_points", "source_slides"],
            },
        },
    },
    "required": ["presentation_title", "slides"],
}


def build_prompt(user_request, retrieved_slides, slide_count):
    # Keep only the fields Gemini needs.
    source_material = [
        {
            "source_file": slide.get("source_file", ""),
            "slide_number": slide.get("slide_number", ""),
            "presentation_title": slide.get("presentation_title", ""),
            "slide_title": slide.get("slide_title", ""),
            "content": slide.get("search_text", ""),
        }
        for slide in retrieved_slides
    ]

    return f"""
You are a presentation content assistant.

Create a presentation based on the user's request and the supplied
source material.

USER REQUEST:
{user_request}

NUMBER OF SLIDES:
Exactly {slide_count} slides.

SOURCE MATERIAL:
{json.dumps(source_material, ensure_ascii=False)}

RULES:
1. Use the supplied source material as the factual basis.
2. Do not invent technical details, results, statistics, or citations.
3. You may reorganize and simplify the information.
4. If the source material does not contain enough information for a
   claim, do not invent it.
5. Make each slide useful and avoid unnecessary repetition.
6. Keep titles concise and bullet points presentation-friendly
   (3 to 6 bullets per slide).
7. For every slide, include the source slides that support its content.
8. Return only valid JSON matching the requested structure.
"""


def generate_slide_content(user_request, retrieved_slides, slide_count):
    prompt = build_prompt(user_request, retrieved_slides, slide_count)

    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.3,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    result = json.loads(response.text)
    generated = result.get("slides", [])

    if len(generated) < slide_count:
        raise ValueError(
            f"Expected {slide_count} slides, received {len(generated)}."
        )

    # If Gemini returns extra slides, keep only the number requested.
    result["slides"] = generated[:slide_count]

    for slide in result["slides"]:
        if not slide.get("title") or not slide.get("key_points"):
            raise ValueError("A generated slide is missing content.")

    return result
