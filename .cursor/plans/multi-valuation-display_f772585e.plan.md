---
name: multi-valuation-display
overview: Change the `/value` flow from “one combined valuation for all images” to “one ValuationResponse per uploaded image”, and update the UI to render a results list aligned with the thumbnail grid.
todos:
  - id: backend-models-and-endpoint
    content: Add multi-response models and update `/value` to accept ordered `image_items` and loop `estimate_value` per image.
    status: completed
  - id: frontend-hidden-fields
    content: Change `index.html` hidden fields to submit `image_items` (JSON per image) preserving order.
    status: completed
  - id: frontend-render-multi-results
    content: Render a list/grid of valuation cards from `payload.results`, aligned to uploaded thumbnails.
    status: completed
  - id: tests-update
    content: Update `/value` tests to the new response shape and assert per-image `estimate_value` call behavior.
    status: completed
isProject: false
---

## Goal

Display **multiple `ValuationResponse`s** (one per uploaded image) instead of a single combined valuation.

## What exists today (why you only see one)

- **Backend**: `[server.py](e:\ai-appraiser\server.py)` `/value` accepts lists (`image_urls`, `image_datas`, `content_types`) but calls `estimate_value(...)` once with *all* images, returning a single `ValuationResponse`.
- **Frontend**: `[index.html](e:\ai-appraiser\index.html)` `htmx:afterRequest` parses the response as a single object and renders one result card.

## Proposed contract (preserve image↔result ordering)

- **Request**: Add ordered form fields `image_items` where each entry is a JSON string representing one uploaded image, e.g.
  - `{ "kind": "gcs", "gcs_uri": "gs://...", "data_url": "data:image/...", "content_type": "image/jpeg" }`
  - `{ "kind": "inline", "data_url": "data:image/...", "content_type": "image/png" }`
  This avoids the ordering ambiguity of separate `image_urls` vs `image_datas` lists when they’re mixed.
- **Response**: Change `/value` to return:
  - `{ "results": [ { "image_index": 0, "valuation": <ValuationResponse> }, ... ] }`

## Backend changes

- Update `[server.py](e:\ai-appraiser\server.py)`
  - **Add models**:
    - `ValuationPerImage { image_index: int, valuation: ValuationResponse }`
    - `MultiValuationResponse { results: list[ValuationPerImage] }`
  - **Update `/value**`:
    - Accept `image_items: list[str] = Form([])`.
    - If `image_items` present: parse JSON per item, decode `data_url` when needed.
    - For each item, call `estimate_value(...)` with exactly one image (`image_uris=[uri]` or `image_data_list=[(bytes,mime)]`).
    - Return `MultiValuationResponse`.
  - **Back-compat** (optional but recommended): if `image_items` is missing, fall back to the old `image_urls/image_datas/content_types` handling, but treat each entry as its own valuation (with best-effort ordering). This keeps API usable for non-updated clients.

## Frontend changes

- Update `[index.html](e:\ai-appraiser\index.html)`
  - In `renderUploadedImages()`:
    - Replace the current hidden-field population of `image_urls`/`image_datas`/`content_types` with `image_items` (one hidden input per image containing JSON).
  - In `htmx:afterRequest`:
    - Parse `payload.results` and render a **grid/list of result cards**, each card showing:
      - thumbnail (from `uploadedImages[image_index].dataUrl`)
      - formatted currency value
      - reasoning
      - sources list

## Test updates

- Update `[test_server.py](e:\ai-appraiser\test_server.py)`
  - Adjust existing `/value` endpoint tests to expect `{"results": [...]}`.
  - Update mocks to verify `estimate_value` is called **N times** (once per image) with one-image inputs.
  - Keep the existing single-image tests, but assert the single result is returned inside `results`.

## Notes / tradeoffs

- This approach makes **2×N Gemini calls** (valuation + parsing per image). It’s the most deterministic mapping of images to responses and produces strict `ValuationResponse` objects per image.

