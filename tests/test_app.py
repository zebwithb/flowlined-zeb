import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from src.app import app
from src.app.services import text_analysis, cache
import numpy as np # Import numpy for cosine similarity calculation if needed in test

client = TestClient(app)

# --- Mock Fixtures ---

@pytest.fixture(autouse=True)
def mock_openai(mocker):
    """Mocks the OpenAI client methods."""
    mock_completion = AsyncMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Mocked Summary"))]
    mocker.patch.object(text_analysis.client.chat.completions, 'create', return_value=mock_completion)

    async def mock_create_embeddings(*args, **kwargs):
        input_data = kwargs.get('input', [])
        if isinstance(input_data, str):
            # Specific query for similarity logic test
            if input_data == "Query for logic test":
                # Use a simple distinct vector
                return MagicMock(data=[MagicMock(embedding=[1.0, 0.0] + [0.0] * 1534)])
            # Default for single string input (e.g., from test_similarity_endpoint)
            elif input_data == "Hello world":
                 return MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
            # Default for other single string inputs
            return MagicMock(data=[MagicMock(embedding=[0.5] * 1536)])

        elif isinstance(input_data, list):
            # Specific texts for similarity logic test
            if input_data == ["Text A", "Text B", "Text C"]:
                # Return distinct vectors for comparison
                return MagicMock(data=[
                    MagicMock(embedding=[0.9, 0.1] + [0.0] * 1534), # High similarity to query [1.0, 0.0]
                    MagicMock(embedding=[0.0, 1.0] + [0.0] * 1534), # Low similarity
                    MagicMock(embedding=[0.5, 0.5] + [0.0] * 1534)  # Medium similarity
                ])
            # Existing logic for len == 3 (used in test_similarity_endpoint)
            elif input_data == ["Hello world", "Goodbye world", "Something completely different"]:
                 return MagicMock(data=[
                    MagicMock(embedding=[0.1] * 1536), # Matches query embedding
                    MagicMock(embedding=[0.2] * 1536),
                    MagicMock(embedding=[0.3] * 1536)
                ])
            # Default for list input
            return MagicMock(data=[MagicMock(embedding=[0.9] * 1536)] * len(input_data))
        return MagicMock(data=[])

    mocker.patch.object(text_analysis.client.embeddings, 'create', side_effect=mock_create_embeddings)

    # Return the mocker instance itself or specific mocks if needed by tests
    # Returning mocker is standard practice if tests don't need direct access to the mocks created here
    return mocker


# --- New Tests ---

def test_similarity_logic(mock_openai, mock_redis_cache):
    """Tests if the similarity endpoint correctly identifies the closest text based on mocked embeddings."""
    query = "Query for logic test"
    texts = ["Text A", "Text B", "Text C"]
    response = client.post(
        "/v1/similarity",
        json={"query": query, "texts": texts}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["closest_text"] == "Text A"

    # Verify mocks were called as expected
    assert text_analysis.client.embeddings.create.call_count == 2 # Once for query, once for texts
    mock_redis_cache["get_embeddings"].assert_called()
    mock_redis_cache["set_embeddings"].assert_called()


def test_summarize_openai_error(mocker, mock_redis_cache):
    """Tests the summarize endpoint response when the OpenAI API call fails."""
    # Override the autouse mock_openai for this specific test
    mocker.patch.object(text_analysis.client.chat.completions, 'create', side_effect=Exception("API Error"))

    response = client.post(
        "/v1/summarize",
        json={"text": "This text summary will fail."}
    )
    assert response.status_code == 500
    assert "API Error" in response.json()["detail"]
    mock_redis_cache["get_summary"].assert_called_once() # Should still check cache first
    mock_redis_cache["set_summary"].assert_not_called() # Should not cache on failure


def test_similarity_openai_error(mocker, mock_redis_cache):
    """Tests the similarity endpoint response when the OpenAI embedding API call fails."""
    # Override the autouse mock_openai for this specific test
    mocker.patch.object(text_analysis.client.embeddings, 'create', side_effect=Exception("Embedding API Error"))

    response = client.post(
        "/v1/similarity",
        json={"query": "Query", "texts": ["Text 1", "Text 2"]}
    )
    assert response.status_code == 500
    assert "Embedding API Error" in response.json()["detail"]
    mock_redis_cache["get_embeddings"].assert_called() # Should still check cache first
    mock_redis_cache["set_embeddings"].assert_not_called() # Should not cache on failure

    return mocker

@pytest.fixture
def mock_redis_cache(mocker):
    """Mocks the RedisLangCache methods."""
    mock_get_summary = AsyncMock(return_value=None)
    mock_set_summary = AsyncMock()
    mock_get_embeddings = AsyncMock(return_value=None)
    mock_set_embeddings = AsyncMock()

    mocker.patch.object(cache.cache, 'get_summary', side_effect=mock_get_summary)
    mocker.patch.object(cache.cache, 'set_summary', side_effect=mock_set_summary)
    mocker.patch.object(cache.cache, 'get_embeddings', side_effect=mock_get_embeddings)
    mocker.patch.object(cache.cache, 'set_embeddings', side_effect=mock_set_embeddings)

    return {
        "get_summary": mock_get_summary,
        "set_summary": mock_set_summary,
        "get_embeddings": mock_get_embeddings,
        "set_embeddings": mock_set_embeddings,
    }

# --- Tests ---

def test_summarize_endpoint(mock_openai, mock_redis_cache):
    response = client.post(
        "/v1/summarize",
        json={"text": "This is a long text that needs to be summarized. It contains multiple sentences and should be reduced to about thirty words while maintaining the key points and meaning of the original text."}
    )
    assert response.status_code == 200
    assert "summary" in response.json()
    assert response.json()["summary"] == "Mocked Summary"
    text_analysis.client.chat.completions.create.assert_called_once()
    mock_redis_cache["set_summary"].assert_called_once()

def test_similarity_endpoint(mock_openai, mock_redis_cache):
    texts_input = ["Hello world", "Goodbye world", "Something completely different"]
    response = client.post(
        "/v1/similarity",
        json={
            "query": "Hello world",
            "texts": texts_input
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "closest_text" in data
    assert "score" in data
    assert text_analysis.client.embeddings.create.call_count == 2
    assert mock_redis_cache["set_embeddings"].call_count == 2

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["uptime"], float)
    assert data["uptime"] >= 0

def test_invalid_input():
    response = client.post("/v1/summarize", json={"text": ""})
    assert response.status_code == 422

    response = client.post("/v1/similarity", json={"query": "", "texts": ["test"]})
    assert response.status_code == 422

    response = client.post("/v1/similarity", json={"query": "test", "texts": []})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_summarize_caching(mock_openai, mock_redis_cache):
    endpoint = "/v1/summarize"
    payload = {"text": "Cache this summary text."}

    response1 = client.post(endpoint, json=payload)
    assert response1.status_code == 200
    summary1 = response1.json()["summary"]
    assert summary1 == "Mocked Summary"
    text_analysis.client.chat.completions.create.assert_called_once()
    mock_redis_cache["get_summary"].assert_called_once()
    mock_redis_cache["set_summary"].assert_called_once()

    text_analysis.client.chat.completions.create.reset_mock()
    mock_redis_cache["get_summary"].return_value = summary1
    mock_redis_cache["get_summary"].reset_mock()

    response2 = client.post(endpoint, json=payload)
    assert response2.status_code == 200
    summary2 = response2.json()["summary"]
    assert summary2 == summary1

    mock_redis_cache["get_summary"].assert_called_once()
    text_analysis.client.chat.completions.create.assert_not_called()
    mock_redis_cache["set_summary"].assert_called_once()

@pytest.mark.asyncio
async def test_embeddings_caching(mock_openai, mock_redis_cache):
    endpoint = "/v1/similarity"
    query = "Cache this query embedding"
    texts = ["Cache this text embedding 1", "Cache this text embedding 2"]
    payload = {"query": query, "texts": texts}

    mock_query_embedding = [0.5] * 1536
    mock_texts_embeddings = [[0.6] * 1536, [0.7] * 1536]

    get_call_count = 0
    async def mock_get_embeddings_side_effect(*args, **kwargs):
        nonlocal get_call_count
        get_call_count += 1
        if get_call_count <= 2:
            return None
        elif get_call_count == 3:
            return mock_query_embedding
        elif get_call_count == 4:
            return mock_texts_embeddings
        return None

    mock_redis_cache["get_embeddings"].side_effect = mock_get_embeddings_side_effect
    mock_redis_cache["get_embeddings"].return_value = None

    response1 = client.post(endpoint, json=payload)
    assert response1.status_code == 200
    assert text_analysis.client.embeddings.create.call_count == 2
    assert mock_redis_cache["get_embeddings"].call_count == 2
    assert mock_redis_cache["set_embeddings"].call_count == 2

    text_analysis.client.embeddings.create.reset_mock()
    mock_redis_cache["set_embeddings"].reset_mock()

    response2 = client.post(endpoint, json=payload)
    assert response2.status_code == 200
    assert response1.json() == response2.json()

    assert mock_redis_cache["get_embeddings"].call_count == 4
    text_analysis.client.embeddings.create.assert_not_called()
    mock_redis_cache["set_embeddings"].assert_not_called()