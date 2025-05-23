# tests/test_api_routes.py
import pytest
from unittest.mock import patch, AsyncMock

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the LLMOps RAG Pipeline API"}

@patch("app.api.routes.rag_service")
def test_query_endpoint(mock_rag_service, client):
    # Mock the async method
    mock_rag_service.process_query = AsyncMock(return_value={
        "answer": "This is a test answer",
        "sources": [{"content": "Test content", "metadata": {"source": "test"}}],
        "metadata": {"model": "test-model"}
    })
    
    response = client.post(
        "/api/query",
        json={"question": "What is RAG?"}
    )
    
    assert response.status_code == 200
    assert response.json()["answer"] == "This is a test answer"

@patch("app.api.routes.rag_service")
def test_index_document_endpoint(mock_rag_service, client):
    mock_rag_service.add_document = AsyncMock(return_value="doc-123")
    
    response = client.post(
        "/api/index-document",
        json={
            "content": "This is a test document",
            "metadata": {"source": "test", "author": "tester"}
        }
    )
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "document_id": "doc-123"}

@patch("app.api.routes.experiment_service")
def test_create_ab_test_endpoint(mock_experiment_service, client):
    mock_experiment_service.create_ab_test.return_value = "test-123"
    
    response = client.post(
        "/api/experiments/exp-123/ab-test",
        json={
            "variant_a": {"template": "Template A", "input_variables": ["query"]},
            "variant_b": {"template": "Template B", "input_variables": ["query"]},
            "evaluation_metric": "accuracy"
        }
    )
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "test_id": "test-123"}
