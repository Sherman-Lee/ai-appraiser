import logging
import os

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)


# --- Configuration ---
load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.5-flash")
STORAGE_BUCKET = None # os.environ.get("STORAGE_BUCKET")
DEFAULT_CURRENCY = os.environ.get(
    "CURRENCY",
    "USD",
)

# The GCS object name limit is 1024 bytes. The timestamp prefix is 26 bytes.
# We truncate the user-provided filename to a safe length to avoid exceeding the limit.
GCS_FILENAME_MAX_LEN = 220


if not STORAGE_BUCKET:
    logging.warning(
        "STORAGE_BUCKET environment variable not set. Image uploads to GCS will be skipped.",
    )