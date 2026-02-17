from __future__ import annotations

import asyncio
import base64
import json
import logging

import pathlib
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.cloud import storage
from typing_extensions import Annotated, Dict, Any
from uuid import uuid4
import time

from data_model import Currency, ValuationPerImage, MultiValuationResponse
from env_config import PROJECT_ID, LOCATION, STORAGE_BUCKET, DEFAULT_CURRENCY
from helpers import upload_image_to_gcs, get_data_url, estimate_value

logging.basicConfig(level=logging.INFO)


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the Google Cloud clients on startup and stores them in the app state."""
    app.state.storage_client = storage.Client(project=PROJECT_ID)
    app.state.client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )
    yield


# --- FastAPI App ---
app = FastAPI(
    title="Item Valuation API",
    description="API to estimate item value based on image and text.",
    lifespan=lifespan,
)


# Serve static files (e.g. script.js) from the same directory as server.py
app.mount("/static", StaticFiles(directory=pathlib.Path(__file__).parent), name="static")


# --- Dependencies ---
def get_storage_client(request: Request) -> storage.Client:
    return request.app.state.storage_client


def get_genai_client(request: Request) -> genai.Client:
    return request.app.state.client


# --- API Endpoints ---
@app.post("/upload-image")
async def upload_image(
    image_file: Annotated[UploadFile, File()],
    storage_client: Annotated[storage.Client, Depends(get_storage_client)],
):
    """Uploads an image, returns a Data URL for preview, and stores the GCS URI."""
    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file type. Please upload an image.",
        )

    try:
        contents = await image_file.read()
        await image_file.seek(0)  # Reset for GCS upload
        image_uri = (
            upload_image_to_gcs(image_file, storage_client) if STORAGE_BUCKET else None
        )
        data_url = get_data_url(image_file, contents)

        return JSONResponse(
            {
                "data_url": data_url,
                "gcs_uri": image_uri,
                "content_type": image_file.content_type,
            },
        )
    except Exception:
        logging.exception("Error uploading image")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while uploading the image.",
        )


# In-memory storage for task progress (use Redis in production)
task_progress: Dict[str, Dict[str, Any]] = {}

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Retrieve progress for a valuation task."""
    if task_id not in task_progress:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(content=task_progress[task_id])


@app.post("/value", response_model=MultiValuationResponse)
async def estimate_item_value(
    description: Annotated[str, Form()],
    image_items: Annotated[list[str], Form()] = [],
    image_urls: Annotated[list[str], Form()] = [],
    image_datas: Annotated[list[str], Form()] = [],
    content_types: Annotated[list[str], Form()] = [],
    image_url: Annotated[str | None, Form()] = None,
    image_data: Annotated[str | None, Form()] = None,
    content_type: Annotated[str | None, Form()] = None,
    currency: Annotated[Currency, Form()] = Currency(DEFAULT_CURRENCY),
    task_id: Annotated[str | None, Form()] = None,
    client: genai.Client = Depends(get_genai_client),
):
    """Estimates the value of an item based on an image and text input."""
    if image_url:
        image_urls = [*image_urls, image_url]
    if image_data:
        image_datas = [*image_datas, image_data]
    if content_type:
        content_types = [*content_types, content_type]

    image_items = [s for s in image_items if s]
    image_urls = [u for u in image_urls if u]
    image_datas = [d for d in image_datas if d]
    content_types = [t for t in content_types if t]

    if not image_items and not image_urls and not image_datas:
        raise HTTPException(
            status_code=400,
            detail="At least one image is required.",
        )

    try:
        # Preferred input: ordered image_items (JSON per image) to preserve UI ordering.
        normalized_items: list[dict] = []
        if image_items:
            for raw in image_items:
                try:
                    item = json.loads(raw)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid image_items JSON.",
                    ) from e
                if not isinstance(item, dict):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid image_items JSON.",
                    )
                normalized_items.append(item)
        else:
            # Back-compat: old separate lists. Order is best-effort when URLs and inline data
            # are mixed, because separate lists lose original ordering.
            normalized_items.extend({"kind": "gcs", "gcs_uri": u} for u in image_urls)
            for i, d in enumerate(image_datas):
                normalized_items.append(
                    {
                        "kind": "inline",
                        "data_url": d,
                        "content_type": (
                            content_types[i]
                            if i < len(content_types) and content_types[i]
                            else None
                        ),
                    },
                )

        # Use provided task_id or generate a new one
        if not task_id:
            task_id = str(uuid4())
        total_images = len(normalized_items)
        task_progress[task_id] = {
            "total": total_images,
            "completed": 0,
            "status": "processing",
        }

        results: list[ValuationPerImage] = []
        for idx, item in enumerate(normalized_items):

            kind = item.get("kind")
            if kind == "gcs":
                gcs_uri = item.get("gcs_uri")
                if not gcs_uri or not isinstance(gcs_uri, str):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid image_items entry: missing gcs_uri.",
                    )
                valuations = estimate_value(
                    image_uris=[gcs_uri],
                    description=description,
                    client=client,
                    image_data_list=None,
                    currency=currency,
                )
            elif kind == "inline":
                data_url = item.get("data_url")
                if not data_url or not isinstance(data_url, str):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid image_items entry: missing data_url.",
                    )
                try:
                    _, encoded_data = data_url.split(",", 1)
                    decoded_image_data = base64.b64decode(encoded_data)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid image_data format.",
                    ) from e

                mime_type = item.get("content_type")
                if mime_type is not None and not isinstance(mime_type, str):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid image_items entry: invalid content_type.",
                    )
                valuations = estimate_value(
                    image_uris=None,
                    description=description,
                    client=client,
                    image_data_list=[(decoded_image_data, mime_type or "image/jpeg")],
                    currency=currency,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image_items entry: unknown kind.",
                )

            results.append(ValuationPerImage(image_index=idx, valuations=valuations))
            
            # Update progress after processing each image
            task_progress[task_id]["completed"] = idx + 1
            task_progress[task_id]["status"] = "processing"
            
            # Small delay to allow progress polling to catch up
            await asyncio.sleep(0.1)

        # Finalize task
        task_progress[task_id]["status"] = "completed"

        response_data = MultiValuationResponse(results=results)
        response = JSONResponse(content=response_data.model_dump())
        response.headers["X-Task-ID"] = task_id
        return response
    except HTTPException:
        raise
    except Exception:
        logging.exception("Internal server error in /value")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during valuation.",
        )


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(pathlib.Path(__file__).parent / "index.html") as f:
        html_content = f.read()
    html_content = html_content.replace(
        "let defaultCurrency = 'USD';",
        f"let defaultCurrency = '{DEFAULT_CURRENCY}';",
    )

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
