from __future__ import annotations

import base64
import io
from unittest.mock import ANY, MagicMock, patch

import pytest
from freezegun import freeze_time
from google.api_core.exceptions import GoogleAPIError
from pydantic import ValidationError

from data_model import ValuationResponse
from server import (
    DEFAULT_CURRENCY,
    Currency,
    estimate_value,
    get_data_url,
    upload_image_to_gcs,
)


def create_mock_gemini_responses(
    valuation_text="Some valuation text",
    parsing_response_text: str | None = None,
    parsing_response_dict: dict | None = None,
    error: Exception | None = None,
) -> list[MagicMock | Exception]:
    """Helper function to create mock Gemini API responses."""
    valuation_mock = MagicMock(
        candidates=[
            MagicMock(content=MagicMock(parts=[MagicMock(text=valuation_text)])),
        ],
    )

    if error:
        return [valuation_mock, error]

    if parsing_response_text:
        parsing_response = MagicMock(text=parsing_response_text)
    elif parsing_response_dict:
        import json

        parsing_response = MagicMock(text=json.dumps(parsing_response_dict))
    else:
        # Default successful parsing response (wrapper with valuations array)
        parsing_response = MagicMock(
            text=f'{{"valuations": [{{"item_name": "some_item", "estimated_value": 100.0, "currency": "{DEFAULT_CURRENCY}", "reasoning": "Looks good", "search_urls": ["example.com"]}}]}}',
        )

    return [valuation_mock, parsing_response]


def test_get_data_url_correct_format() -> None:
    # Create a custom mock for UploadFile
    file_content = b"fake image content"
    mock_file = MagicMock()
    mock_file.filename = "test.jpg"
    mock_file.content_type = "image/jpeg"
    mock_file.read.return_value = file_content
    contents = file_content
    data_url = get_data_url(mock_file, contents)
    assert data_url == "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50"


def test_estimate_value_image_uri_success_eur(
    mock_google_cloud_clients_and_app,
) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_models = mock_genai_client.models
    mock_models.generate_content.side_effect = create_mock_gemini_responses(
        parsing_response_dict={
            "valuations": [
                {
                    "item_name": "some_item",
                    "estimated_value": 100.0,
                    "currency": "EUR",
                    "reasoning": "Looks good",
                    "search_urls": ["example.com"],
                },
            ],
        },
    )

    response = estimate_value(
        image_uris=["gs://some_bucket/some_image.jpg"],
        description="A test item",
        currency=Currency.EUR,
        client=mock_genai_client,
    )

    assert len(response) == 1
    assert response[0].estimated_value == 100.0
    assert response[0].currency == Currency.EUR
    assert response[0].reasoning == "Looks good"
    assert response[0].search_urls == ["example.com"]
    assert mock_models.generate_content.call_count == 2


def test_estimate_value_image_data_success(mock_google_cloud_clients_and_app) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_models = mock_genai_client.models
    mock_models.generate_content.side_effect = create_mock_gemini_responses()

    image_data = b"fake image data"
    response = estimate_value(
        image_uris=None,
        description="Test item with data",
        image_data_list=[(image_data, "image/jpeg")],
        client=mock_genai_client,
    )

    assert len(response) == 1
    assert response[0].estimated_value == 100.0
    assert response[0].currency == Currency(DEFAULT_CURRENCY)
    assert response[0].reasoning == "Looks good"
    assert response[0].search_urls == ["example.com"]
    assert mock_models.generate_content.call_count == 2


def test_estimate_value_raises_exception_no_image() -> None:
    """Tests that estimate_value raises a ValueError when no image is provided."""
    with pytest.raises(ValueError, match="Must provide at least one image"):
        estimate_value(
            image_uris=None,
            description="Test",
            image_data_list=None,
            client=MagicMock(),
        )


def test_estimate_value_valuation_api_error(mock_google_cloud_clients_and_app) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_genai_client.models.generate_content.side_effect = GoogleAPIError(
        "Gemini API error",
    )

    with pytest.raises(GoogleAPIError) as exc_info:
        estimate_value(
            image_uris=["gs://some_bucket/some_image.jpg"],
            description="A test item",
            client=mock_genai_client,
        )
    assert str(exc_info.value) == "Gemini API error"


def test_estimate_value_parsing_api_error(mock_google_cloud_clients_and_app) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_genai_client.models.generate_content.side_effect = (
        create_mock_gemini_responses(
            error=GoogleAPIError("Gemini API error during parsing"),
        )
    )

    with pytest.raises(GoogleAPIError) as exc_info:
        estimate_value(
            image_uris=["gs://some_bucket/some_image.jpg"],
            description="A test item",
            client=mock_genai_client,
        )
    assert str(exc_info.value) == "Gemini API error during parsing"


def test_estimate_value_malformed_json_response(
    mock_google_cloud_clients_and_app,
) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_genai_client.models.generate_content.side_effect = (
        create_mock_gemini_responses(
            parsing_response_text='{"valuations": [{"wrong_field": "some value", "currency": "USD"}]}',
        )
    )

    with pytest.raises(ValidationError):
        estimate_value(
            image_uris=["gs://some_bucket/some_image.jpg"],
            description="A test item",
            client=mock_genai_client,
        )

@pytest.mark.parametrize("parsing_text,description", [
    ('{"valuations": [{"estimated_value": 100.0, "currency": "USD", "reasoning": "Looks good", "search_urls": "not-a-list"}]}', "invalid search_urls"),
    ('{"valuations": [{"item_name": "some_item", "estimated_value": "not-a-number", "currency": "USD", "reasoning": "Looks good", "search_urls": ["example.com"]}]}', "invalid estimated_value"),
])
def test_estimate_value_invalid_fields(
    parsing_text, description, mock_google_cloud_clients_and_app,
) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_genai_client.models.generate_content.side_effect = create_mock_gemini_responses(
        parsing_response_text=parsing_text,
    )

    with pytest.raises(ValidationError):
        estimate_value(
            image_uris=["gs://some_bucket/some_image.jpg"],
            description="A test item",
            client=mock_genai_client,
        )


def test_estimate_value_returns_multiple_valuations(
    mock_google_cloud_clients_and_app,
) -> None:
    """When parsing returns multiple ValuationResponse objects, estimate_value returns a list of all."""
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_models = mock_genai_client.models
    mock_models.generate_content.side_effect = create_mock_gemini_responses(
        parsing_response_dict={
            "valuations": [
                {
                    "item_name": "some_item",
                    "estimated_value": 50.0,
                    "currency": "USD",
                    "reasoning": "First item",
                    "search_urls": ["http://a.com"],
                },
                {
                    "item_name": "some_item",
                    "estimated_value": 75.0,
                    "currency": "USD",
                    "reasoning": "Second item",
                    "search_urls": ["http://b.com"],
                },
            ],
        },
    )

    response = estimate_value(
        image_uris=["gs://some_bucket/some_image.jpg"],
        description="Multiple items in one image",
        client=mock_genai_client,
    )

    assert len(response) == 2
    assert response[0].estimated_value == 50.0
    assert response[0].reasoning == "First item"
    assert response[1].estimated_value == 75.0
    assert response[1].reasoning == "Second item"
    assert mock_models.generate_content.call_count == 2


def test_estimate_value_multiple_image_uris_sends_multiple_parts(
    mock_google_cloud_clients_and_app,
) -> None:
    _, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_models = mock_genai_client.models
    mock_models.generate_content.side_effect = create_mock_gemini_responses()

    estimate_value(
        image_uris=[
            "gs://some_bucket/some_image_1.jpg",
            "gs://some_bucket/some_image_2.png",
        ],
        description="A test item",
        client=mock_genai_client,
    )

    assert mock_models.generate_content.call_count == 2
    first_call_kwargs = mock_models.generate_content.call_args_list[0].kwargs
    contents = first_call_kwargs["contents"]
    assert len(contents) == 3
    assert isinstance(contents[-1], str)
    assert contents[-1].startswith("You are a professional appraiser")


@freeze_time("2023-01-01 12:00:00")
def test_upload_image_to_gcs(mock_google_cloud_clients_and_app) -> None:
    """Tests the image upload functionality to Google Cloud Storage."""
    _, mock_storage_client, _ = mock_google_cloud_clients_and_app
    mock_bucket = mock_storage_client.bucket.return_value
    mock_bucket.name = "test-bucket"
    mock_blob = mock_bucket.blob.return_value

    with patch("helpers.STORAGE_BUCKET", "test-bucket"):
        file_content = b"fake image content"
        mock_file = MagicMock()
        mock_file.filename = "test.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.file = io.BytesIO(file_content)

        gcs_uri = upload_image_to_gcs(mock_file, mock_storage_client)

        expected_filename = "20230101120000000000_test.jpg"
        expected_uri = f"gs://{mock_bucket.name}/{expected_filename}"

        assert gcs_uri == expected_uri
        mock_storage_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with(expected_filename)
        mock_blob.upload_from_file.assert_called_once_with(
            mock_file.file,
            content_type="image/jpeg",
        )


@patch("server.STORAGE_BUCKET", "test-bucket")
@patch("server.upload_image_to_gcs")
def test_upload_image_endpoint_success_with_gcs(
    mock_upload_image_to_gcs,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_upload_image_to_gcs.return_value = "gs://test-bucket/test.jpg"
    test_image_content = b"fake image content"
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.jpg", test_image_content, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json() == {
        "data_url": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
        "gcs_uri": "gs://test-bucket/test.jpg",
        "content_type": "image/jpeg",
    }
    mock_upload_image_to_gcs.assert_called_once()


@patch("server.STORAGE_BUCKET", None)
@patch("server.upload_image_to_gcs")  # Still mock to ensure it's NOT called
def test_upload_image_no_storage_bucket(
    mock_upload_image_to_gcs_not_called,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    test_image_content = b"fake image content"
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.jpg", test_image_content, "image/jpeg")},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["gcs_uri"] is None, (
        "gcs_uri should be None when STORAGE_BUCKET is not set"
    )
    assert (
        response_json["data_url"] == "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50"
    )
    assert response_json["content_type"] == "image/jpeg"
    mock_upload_image_to_gcs_not_called.assert_not_called()


@patch("server.STORAGE_BUCKET", "test-bucket")  # Ensure STORAGE_BUCKET is set
@patch("server.upload_image_to_gcs")
def test_upload_image_gcs_upload_fails(
    mock_upload_image_to_gcs,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_upload_image_to_gcs.side_effect = Exception("GCS upload failed")
    test_image_content = b"fake image content"
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.jpg", test_image_content, "image/jpeg")},
    )
    assert response.status_code == 500
    assert response.json() == {
        "detail": "An error occurred while uploading the image.",
    }
    mock_upload_image_to_gcs.assert_called_once()


def test_upload_image_invalid_type(mock_google_cloud_clients_and_app) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    response = client.post(
        "/upload-image",
        files={"image_file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid image file type. Please upload an image.",
    }


@patch("server.estimate_value")
def test_value_endpoint_success_gbp(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.return_value = [
        ValuationResponse(
            item_name="some_item",
            estimated_value=123.45,
            currency=Currency.GBP,
            reasoning="Looks nice",
            search_urls=["http://example.com"],
        ),
    ]
    response = client.post(
        "/value",
        data={
            "description": "A test item",
            "image_datas": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_types": "image/jpeg",
            "currency": "GBP",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "image_index": 0,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 123.45,
                        "currency": "GBP",
                        "reasoning": "Looks nice",
                        "search_urls": ["http://example.com"],
                    },
                ],
            },
        ],
    }
    mock_estimate_value.assert_called_once_with(
        image_uris=None,
        description="A test item",
        client=ANY,
        image_data_list=[(b"fake image content", "image/jpeg")],
        currency=Currency.GBP,
    )


@patch("server.estimate_value")
def test_value_endpoint_success_image_url(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.return_value = [
        ValuationResponse(
            item_name="some_item",
            estimated_value=123.45,
            currency=Currency.USD,
            reasoning="Looks nice from URL",
            search_urls=["http://example.com/url_image"],
        ),
    ]
    response = client.post(
        "/value",
        data={
            "description": "A test item from URL",
            "image_urls": "gs://test-bucket/test_image.jpg",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "image_index": 0,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 123.45,
                        "currency": "USD",
                        "reasoning": "Looks nice from URL",
                        "search_urls": ["http://example.com/url_image"],
                    },
                ],
            },
        ],
    }
    mock_estimate_value.assert_called_once_with(
        image_uris=["gs://test-bucket/test_image.jpg"],
        description="A test item from URL",
        client=ANY,
        image_data_list=None,
        currency=Currency(DEFAULT_CURRENCY),
    )


@patch("server.estimate_value")
def test_value_endpoint_multiple_image_urls(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.return_value = [
        ValuationResponse(
            item_name="some_item",
            estimated_value=10.0,
            currency=Currency.USD,
            reasoning="Multiple URLs",
            search_urls=[],
        ),
    ]

    response = client.post(
        "/value",
        data={
            "description": "A test item with multiple URLs",
            "image_urls": [
                "gs://test-bucket/image_1.jpg",
                "gs://test-bucket/image_2.jpg",
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "image_index": 0,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 10.0,
                        "currency": "USD",
                        "reasoning": "Multiple URLs",
                        "search_urls": [],
                    },
                ],
            },
            {
                "image_index": 1,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 10.0,
                        "currency": "USD",
                        "reasoning": "Multiple URLs",
                        "search_urls": [],
                    },
                ],
            },
        ],
    }
    assert mock_estimate_value.call_count == 2
    mock_estimate_value.assert_any_call(
        image_uris=["gs://test-bucket/image_1.jpg"],
        description="A test item with multiple URLs",
        client=ANY,
        image_data_list=None,
        currency=Currency(DEFAULT_CURRENCY),
    )
    mock_estimate_value.assert_any_call(
        image_uris=["gs://test-bucket/image_2.jpg"],
        description="A test item with multiple URLs",
        client=ANY,
        image_data_list=None,
        currency=Currency(DEFAULT_CURRENCY),
    )


@patch("server.estimate_value")
def test_value_endpoint_multiple_image_datas(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.return_value = [
        ValuationResponse(
            item_name="some_item",
            estimated_value=11.0,
            currency=Currency.USD,
            reasoning="Multiple inline images",
            search_urls=[],
        ),
    ]

    response = client.post(
        "/value",
        data={
            "description": "A test item with multiple inline images",
            "image_datas": [
                "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
                "data:image/png;base64,ZmFrZSBpbWFnZSBkYXRh",
            ],
            "content_types": ["image/jpeg", "image/png"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "image_index": 0,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 11.0,
                        "currency": "USD",
                        "reasoning": "Multiple inline images",
                        "search_urls": [],
                    },
                ],
            },
            {
                "image_index": 1,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 11.0,
                        "currency": "USD",
                        "reasoning": "Multiple inline images",
                        "search_urls": [],
                    },
                ],
            },
        ],
    }
    assert mock_estimate_value.call_count == 2
    mock_estimate_value.assert_any_call(
        image_uris=None,
        description="A test item with multiple inline images",
        client=ANY,
        image_data_list=[(b"fake image content", "image/jpeg")],
        currency=Currency(DEFAULT_CURRENCY),
    )
    mock_estimate_value.assert_any_call(
        image_uris=None,
        description="A test item with multiple inline images",
        client=ANY,
        image_data_list=[(b"fake image data", "image/png")],
        currency=Currency(DEFAULT_CURRENCY),
    )


@patch("server.estimate_value")
def test_value_endpoint_mixed_image_urls_and_datas(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.return_value = [
        ValuationResponse(
            item_name="some_item",
            estimated_value=12.0,
            currency=Currency.USD,
            reasoning="Mixed images",
            search_urls=[],
        ),
    ]

    response = client.post(
        "/value",
        data={
            "description": "A test item with mixed images",
            "image_urls": ["gs://test-bucket/url_image.jpg"],
            "image_datas": ["data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50"],
            "content_types": ["image/jpeg"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "image_index": 0,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 12.0,
                        "currency": "USD",
                        "reasoning": "Mixed images",
                        "search_urls": [],
                    },
                ],
            },
            {
                "image_index": 1,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 12.0,
                        "currency": "USD",
                        "reasoning": "Mixed images",
                        "search_urls": [],
                    },
                ],
            },
        ],
    }
    assert mock_estimate_value.call_count == 2
    mock_estimate_value.assert_any_call(
        image_uris=["gs://test-bucket/url_image.jpg"],
        description="A test item with mixed images",
        client=ANY,
        image_data_list=None,
        currency=Currency(DEFAULT_CURRENCY),
    )
    mock_estimate_value.assert_any_call(
        image_uris=None,
        description="A test item with mixed images",
        client=ANY,
        image_data_list=[(b"fake image content", "image/jpeg")],
        currency=Currency(DEFAULT_CURRENCY),
    )


@patch("server.estimate_value")
def test_value_endpoint_uses_image_data_when_url_is_empty(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.return_value = [
        ValuationResponse(
            item_name="some_item",
            estimated_value=50.0,
            currency=Currency.CAD,
            reasoning="Data with empty URL",
            search_urls=[],
        ),
    ]
    response = client.post(
        "/value",
        data={
            "description": "A test item with empty URL",
            "image_urls": "",
            "image_datas": "data:image/png;base64,ZmFrZSBpbWFnZSBkYXRh",
            "content_types": "image/png",
            "currency": "CAD",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "image_index": 0,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 50.0,
                        "currency": "CAD",
                        "reasoning": "Data with empty URL",
                        "search_urls": [],
                    },
                ],
            },
        ],
    }
    mock_estimate_value.assert_called_once_with(
        image_uris=None,
        description="A test item with empty URL",
        client=ANY,
        image_data_list=[(b"fake image data", "image/png")],
        currency=Currency.CAD,
    )


@patch("server.estimate_value")
def test_value_endpoint_both_inputs_prioritizes_url(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.return_value = [
        ValuationResponse(
            item_name="some_item",
            estimated_value=200.0,
            currency=Currency.JPY,
            reasoning="URL should be prioritized",
            search_urls=["http://example.com/both"],
        ),
    ]
    image_data_str = (
        "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    )
    response = client.post(
        "/value",
        data={
            "description": "A test item with both URL and data",
            "image_urls": "gs://test-bucket/preferred_image.jpg",
            "image_datas": image_data_str,
            "content_types": "image/gif",
            "currency": "JPY",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "image_index": 0,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 200.0,
                        "currency": "JPY",
                        "reasoning": "URL should be prioritized",
                        "search_urls": ["http://example.com/both"],
                    },
                ],
            },
            {
                "image_index": 1,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 200.0,
                        "currency": "JPY",
                        "reasoning": "URL should be prioritized",
                        "search_urls": ["http://example.com/both"],
                    },
                ],
            },
        ],
    }
    assert mock_estimate_value.call_count == 2
    mock_estimate_value.assert_any_call(
        image_uris=["gs://test-bucket/preferred_image.jpg"],
        description="A test item with both URL and data",
        client=ANY,
        image_data_list=None,
        currency=Currency.JPY,
    )
    mock_estimate_value.assert_any_call(
        image_uris=None,
        description="A test item with both URL and data",
        client=ANY,
        image_data_list=[
            (base64.b64decode(image_data_str.split(",", 1)[1]), "image/gif"),
        ],
        currency=Currency.JPY,
    )


@patch("server.estimate_value")
def test_value_endpoint_one_image_two_valuations(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    """One image can return multiple valuations (e.g. multiple items in one photo)."""
    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.return_value = [
        ValuationResponse(
            item_name="some_item",
            estimated_value=30.0,
            currency=Currency.USD,
            reasoning="Item A",
            search_urls=[],
        ),
        ValuationResponse(
            item_name="some_item",
            estimated_value=45.0,
            currency=Currency.USD,
            reasoning="Item B",
            search_urls=["http://example.com"],
        ),
    ]

    response = client.post(
        "/value",
        data={
            "description": "Two items in one image",
            "image_datas": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_types": "image/jpeg",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["image_index"] == 0
    valuations = data["results"][0]["valuations"]
    assert len(valuations) == 2
    assert valuations[0]["estimated_value"] == 30.0
    assert valuations[0]["reasoning"] == "Item A"
    assert valuations[1]["estimated_value"] == 45.0
    assert valuations[1]["reasoning"] == "Item B"
    assert valuations[1]["search_urls"] == ["http://example.com"]
    mock_estimate_value.assert_called_once()


@patch("server.estimate_value")
def test_value_endpoint_estimate_value_exception(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.side_effect = Exception("Something went wrong")
    response = client.post(
        "/value",
        data={
            "description": "A test item that causes an error",
            "image_datas": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_types": "image/jpeg",
        },
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "An internal error occurred during valuation."}
    mock_estimate_value.assert_called_once()


def test_value_endpoint_no_image_provided(mock_google_cloud_clients_and_app) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    response = client.post("/value", data={"description": "A test item"})
    assert response.status_code == 400
    assert response.json() == {"detail": "At least one image is required."}


def _assert_html_contains_currency(response, currency) -> None:
    """Helper to check for currency in the root HTML response."""
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert f"let defaultCurrency = '{currency}';" in response.text


def test_read_root_serves_html_with_default_currency(
    mock_google_cloud_clients_and_app,
) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    with patch("server.DEFAULT_CURRENCY", "XYZ"):
        response = client.get("/")
        _assert_html_contains_currency(response, "XYZ")


def test_health_check(mock_google_cloud_clients_and_app) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_value_endpoint_invalid_currency(mock_google_cloud_clients_and_app) -> None:
    client, _, _ = mock_google_cloud_clients_and_app
    response = client.post(
        "/value",
        data={
            "description": "A test item",
            "image_datas": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_types": "image/jpeg",
            "currency": "INVALID_CURRENCY",
        },
    )
    assert (
        response.status_code == 422
    )  # Unprocessable Entity for Pydantic validation error
    assert "Input should be 'USD', 'EUR', 'GBP', 'JPY' or 'CAD'" in response.text


def test_value_endpoint_integration_style(mock_google_cloud_clients_and_app) -> None:
    """Tests the /value endpoint all the way down to the Gemini client mock,
    without mocking the intermediate estimate_value function.
    """
    client, _, mock_genai_client = mock_google_cloud_clients_and_app
    mock_models = mock_genai_client.models
    mock_models.generate_content.side_effect = create_mock_gemini_responses(
        parsing_response_dict={
            "valuations": [
                {
                    "item_name": "some_item",
                    "estimated_value": 99.99,
                    "currency": "USD",
                    "reasoning": "Integration test success",
                    "search_urls": [],
                },
            ],
        },
    )

    response = client.post(
        "/value",
        data={
            "description": "An integration test item",
            "image_datas": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_types": "image/jpeg",
            "currency": "USD",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "image_index": 0,
                "valuations": [
                    {
                        "item_name": "some_item",
                        "estimated_value": 99.99,
                        "currency": "USD",
                        "reasoning": "Integration test success",
                        "search_urls": [],
                    },
                ],
            },
        ],
    }
    assert mock_models.generate_content.call_count == 2


def test_progress_endpoint_returns_current_step(mock_google_cloud_clients_and_app) -> None:
    """GET /progress/{task_id} should include the current_step field."""
    from server import task_progress, get_progress_lock
    import asyncio

    client, _, _ = mock_google_cloud_clients_and_app
    test_task_id = "test_progress_step"

    # Manually seed a progress entry
    task_progress[test_task_id] = {
        "total": 3,
        "completed": 1,
        "status": "processing",
        "current_step": "Appraised image 1 of 3...",
    }

    response = client.get(f"/progress/{test_task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["current_step"] == "Appraised image 1 of 3..."
    assert data["completed"] == 1
    assert data["total"] == 3

    # Clean up
    del task_progress[test_task_id]


@patch("server.estimate_value")
def test_progress_updated_during_valuation(
    mock_estimate_value,
    mock_google_cloud_clients_and_app,
) -> None:
    """Progress dict should contain current_step after valuation completes."""
    from server import task_progress

    client, _, _ = mock_google_cloud_clients_and_app
    mock_estimate_value.return_value = [
        ValuationResponse(
            item_name="some_item",
            estimated_value=50.0,
            currency=Currency.USD,
            reasoning="Progress test",
            search_urls=[],
        ),
    ]

    response = client.post(
        "/value",
        data={
            "description": "Progress test item",
            "image_datas": "data:image/jpeg;base64,ZmFrZSBpbWFnZSBjb250ZW50",
            "content_types": "image/jpeg",
            "task_id": "test_progress_valuation",
        },
    )
    assert response.status_code == 200

    # After completion, the progress entry should exist with current_step set
    progress = task_progress.get("test_progress_valuation")
    assert progress is not None
    assert progress["status"] == "completed"
    assert progress["current_step"] == "Complete!"
    assert progress["completed"] == progress["total"]

    # Clean up
    if "test_progress_valuation" in task_progress:
        del task_progress["test_progress_valuation"]
