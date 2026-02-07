---
name: Option B multiple valuations per image
overview: "Implement Option B: support multiple ValuationResponse objects per image by changing the API to use valuations (list) per image, updating estimate_value to parse and return a list, and rendering multiple valuation cards or sub-items per image in the UI."
todos: []
isProject: false
---

# Option B: Multiple valuations per image

## Goal

When the parsing model returns an **array** of ValuationResponse objects (e.g. one image shows multiple items), expose all of them: one `ValuationPerImage` per image with a **list** of valuations, and render each valuation in the UI (e.g. "Image 1 – Item 1: $X, Item 2: $Y").

## Contract change

- **Current**: `ValuationPerImage { image_index: int, valuation: ValuationResponse }` (one valuation per image).
- **New**: `ValuationPerImage { image_index: int, valuations: list[ValuationResponse] }` (one image, zero or more valuations).

Back-compat: Frontend can treat legacy `valuation` (singular) as `valuations: [valuation]` when present.

---

## 1. Data model ([data_model.py](e:\ai-appraiser\data_model.py))

- Change `ValuationPerImage` from `valuation: ValuationResponse` to `valuations: list[ValuationResponse]`.
- Add a parsing wrapper for the Gemini response (so the schema asks for an array):
  - e.g. `class ValuationParsingResult(BaseModel): valuations: list[ValuationResponse]` (used only in helpers for `response_schema` and parsing).

---

## 2. Parsing and return type ([helpers.py](e:\ai-appraiser\helpers.py))

- **Parsing**:
  - Use a schema that expects an array: pass `ValuationParsingResult` (or equivalent) as `response_schema` for the parsing Gemini call so the model returns `{ "valuations": [ ... ] }`.
  - After getting `valuation_response_text`, parse JSON once. If the root is an array, validate each element with `ValuationResponse.model_validate(obj)` and build the list. If the root is a single object, wrap it in a one-element list. (Handles both array and single-object responses.)
- **Return type**: Change `estimate_value` to return `list[ValuationResponse]` instead of `ValuationResponse`.
- **Currency**: Apply the existing currency warning per item (or once for the list) as appropriate.

---

## 3. Server ([server.py](e:\ai-appraiser\server.py))

- For each image, call `estimate_value(...)` and receive `list[ValuationResponse]` (e.g. `valuations = estimate_value(...)`).
- Build one `ValuationPerImage` per image with the full list: `ValuationPerImage(image_index=idx, valuations=valuations)`.
- Ensure the response is still `MultiValuationResponse(results=results)`; each result now has `valuations` (list) instead of `valuation` (single).

---

## 4. Frontend ([index.html](e:\ai-appraiser\index.html))

- In the `/value` response handler (around lines 282–339):
  - Normalize each result: if `entry.valuation` exists (back-compat), treat as `valuations = [entry.valuation]`; otherwise use `entry.valuations` (ensure it’s an array, default to `[]`).
  - For each result entry, keep one card per **image** (thumbnail + "Image N").
  - Inside that card, iterate over `valuations` and render a **sub-block per valuation** (e.g. "Estimated Value: $X", "Reasoning: ...", "Sources: ..." for each item). Optionally label as "Item 1", "Item 2" if multiple.
- Keep existing formatting (currency, reasoning, search_urls) per valuation.

---

## 5. Tests ([test_server.py](e:\ai-appraiser\test_server.py))

- **Unit tests for `estimate_value**`: Update tests that assert return type to expect `list[ValuationResponse]` (e.g. single-element list when model returns one object). Add a test that mocks parsing to return multiple objects and asserts a multi-element list.
- **Integration/endpoint tests**: Update mocks from `mock_estimate_value.return_value = ValuationResponse(...)` to `mock_estimate_value.return_value = [ValuationResponse(...)]`. Change assertions from `response["results"][i]["valuation"]` to `response["results"][i]["valuations"]` (list). Add a test where one image returns two valuations and assert `results[0]["valuations"]` has length 2 and correct structure.

---

## Summary


| File                                             | Change                                                                                                                     |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| [data_model.py](e:\ai-appraiser\data_model.py)   | `ValuationPerImage.valuations: list[ValuationResponse]`; add `ValuationParsingResult` (or equivalent) for parsing.         |
| [helpers.py](e:\ai-appraiser\helpers.py)         | Parse single or array into `list[ValuationResponse]`; return `list[ValuationResponse]`; use array schema for parsing call. |
| [server.py](e:\ai-appraiser\server.py)           | Use `valuations = estimate_value(...)` and `ValuationPerImage(image_index=idx, valuations=valuations)`.                    |
| [index.html](e:\ai-appraiser\index.html)         | Normalize `valuation` → `valuations`; render multiple valuation sub-blocks per image.                                      |
| [test_server.py](e:\ai-appraiser\test_server.py) | Mocks return lists; assert `valuations` array; add multi-valuation test.                                                   |


