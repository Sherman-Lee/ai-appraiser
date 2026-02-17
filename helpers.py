from fastapi import UploadFile
from google.cloud import storage
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Part, Tool
from werkzeug.utils import secure_filename
from mimetypes import guess_type
from io import BytesIO

import base64
import datetime
import json
import logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("Pillow not available, image compression disabled")

from data_model import Currency, ValuationParsingResult, ValuationResponse
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


def compress_image_for_preview(contents: bytes, content_type: str, max_size: int = 800, quality: int = 85) -> tuple[bytes, str]:
    """
    Compress an image for preview thumbnail while maintaining aspect ratio.
    Returns (compressed_bytes, mime_type).
    Falls back to original if compression fails or PIL is unavailable.
    """
    if not PIL_AVAILABLE:
        return contents, content_type
    
    try:
        # Open image from bytes
        img = Image.open(BytesIO(contents))
        
        # Convert RGBA to RGB if needed (for JPEG compatibility)
        if img.mode in ("RGBA", "LA", "P"):
            # Create white background for transparency
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = rgb_img
        elif img.mode != "RGB":
            img = img.convert("RGB")
        
        # Calculate new dimensions maintaining aspect ratio
        width, height = img.size
        if width <= max_size and height <= max_size:
            # Image is already small enough, but still compress for size reduction
            pass
        else:
            # Resize maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save compressed image to bytes
        output = BytesIO()
        # Use JPEG for compressed output (smaller file size)
        img.save(output, format="JPEG", quality=quality, optimize=True)
        compressed_bytes = output.getvalue()
        
        return compressed_bytes, "image/jpeg"
    except Exception as e:
        logging.warning(f"Failed to compress image for preview: {e}, using original")
        return contents, content_type


def get_data_url(file: UploadFile, contents: bytes, compress: bool = True) -> str:
    """
    Creates a data URL for the image preview.
    If compress=True, compresses the image for smaller preview size.
    Original image is still uploaded to GCS for full-quality valuation.
    """
    if compress:
        compressed_contents, mime_type = compress_image_for_preview(contents, file.content_type)
        encoded_image = base64.b64encode(compressed_contents).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_image}"
    else:
        encoded_image = base64.b64encode(contents).decode("utf-8")
        return f"data:{file.content_type};base64,{encoded_image}"


def estimate_value(
    *,
    image_uris: list[str] | None,
    description: str,
    client: genai.Client,
    image_data_list: list[tuple[bytes, str]] | None = None,
    currency: Currency = Currency(DEFAULT_CURRENCY),
) -> list[ValuationResponse]:
    """Calls Gemini API with Search Tool to estimate item value, then parses the result into a list of ValuationResponse."""
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

    # Second Gemini call to parse the valuation string into a list of ValuationResponse
    config_for_parsing = GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ValuationParsingResult,
    )

    parsing_prompt = PARSING_PROMPT.format(
        valuation_text,
        ValuationParsingResult.model_json_schema(),
        currency.value,
    )
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

    parsed = json.loads(valuation_response_text)
    if isinstance(parsed, list):
        validated_list = [ValuationResponse.model_validate(obj) for obj in parsed]
    elif isinstance(parsed, dict) and "valuations" in parsed:
        validated_list = [
            ValuationResponse.model_validate(obj) for obj in parsed["valuations"]
        ]
    else:
        validated_list = [ValuationResponse.model_validate(parsed)]

    for v in validated_list:
        if v.currency != currency:
            logging.warning(
                "Model returned currency '%s' but '%s' was requested.",
                v.currency.value,
                currency.value,
            )
    return validated_list