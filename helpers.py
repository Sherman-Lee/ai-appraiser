from fastapi import UploadFile
from google.cloud import storage
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Part, Tool
from werkzeug.utils import secure_filename
from mimetypes import guess_type


import base64
import datetime
import logging

from data_model import Currency, ValuationResponse
from env_config import STORAGE_BUCKET, GCS_FILENAME_MAX_LEN, DEFAULT_CURRENCY, MODEL_ID
from prompts import VALUATION_PROMPT, PARSING_PROMPT

logging.basicConfig(level=logging.INFO)


# --- Helper Functions ---
def upload_image_to_gcs(file: UploadFile, storage_client: storage.Client) -> str:
    """Uploads an image file to Google Cloud Storage and returns the GCS URI."""
    assert storage_client is not None
    bucket = storage_client.bucket(STORAGE_BUCKET)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S%f")
    # Ensure filename is not None before securing it
    filename_to_secure = file.filename or "unknown_file"
    safe_filename = secure_filename(filename_to_secure)
    # Truncate filename to prevent object names from exceeding GCS limits.
    truncated_filename = safe_filename[:GCS_FILENAME_MAX_LEN]
    filename = f"{timestamp}_{truncated_filename}"
    blob = bucket.blob(filename)

    try:
        blob.upload_from_file(file.file, content_type=file.content_type)
        return f"gs://{STORAGE_BUCKET}/{filename}"
    except Exception:
        logging.exception("Error uploading image to Cloud Storage")
        raise


def get_data_url(file: UploadFile, contents: bytes) -> str:
    """Creates a data URL for the image."""
    encoded_image = base64.b64encode(contents).decode("utf-8")
    return f"data:{file.content_type};base64,{encoded_image}"


def estimate_value(
    *,
    image_uris: list[str] | None,
    description: str,
    client: genai.Client,
    image_data_list: list[tuple[bytes, str]] | None = None,
    currency: Currency = Currency(DEFAULT_CURRENCY),
) -> ValuationResponse:
    """Calls Gemini API with Search Tool to estimate item value, then parses the result into a ValuationResponse."""
    assert client is not None
    valuation_prompt = VALUATION_PROMPT.format(description, currency.value)
    config_with_search = GenerateContentConfig(
        tools=[Tool(google_search=GoogleSearch())],
    )

    image_parts: list[Part] = []
    if image_uris:
        image_parts.extend(
            Part.from_uri(
                file_uri=image_uri,
                mime_type=guess_type(image_uri)[0] or "image/jpeg",
            )
            for image_uri in image_uris
        )
    if image_data_list:
        image_parts.extend(
            Part.from_bytes(
                data=image_data,
                mime_type=mime_type or "image/jpeg",
            )
            for (image_data, mime_type) in image_data_list
        )

    if not image_parts:
        msg = "Must provide at least one image"
        raise ValueError(msg)

    response_with_search = client.models.generate_content(
        model=MODEL_ID,
        contents=[*image_parts, valuation_prompt],
        config=config_with_search,
    )

    # Use final part of search results with answer
    valuation_text = None
    if (
        response_with_search
        and response_with_search.candidates
        and response_with_search.candidates[0].content
        and response_with_search.candidates[0].content.parts
    ):
        for part in response_with_search.candidates[0].content.parts:
            if part.text:
                valuation_text = part.text
                break

    if not valuation_text:
        msg = "Failed to get a text response from the valuation model."
        raise ValueError(msg)

    # Second Gemini call to parse the valuation string into a ValuationResponse
    config_for_parsing = GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ValuationResponse,
    )
    
    parsing_prompt = PARSING_PROMPT.format(valuation_text, ValuationResponse.model_json_schema(), currency.value)
    response_for_parsing = client.models.generate_content(
        model=MODEL_ID,
        contents=parsing_prompt,
        config=config_for_parsing,
    )
    valuation_response_text = (
        response_for_parsing.text if response_for_parsing else None
    )
    if not valuation_response_text:
        msg = "Failed to get a valid JSON response from the parsing model."
        raise ValueError(msg)

    validated_response = ValuationResponse.model_validate_json(valuation_response_text)
    if validated_response.currency != currency:
        logging.warning(
            "Model returned currency '%s' but '%s' was requested.",
            validated_response.currency.value,
            currency.value,
        )
    return validated_response