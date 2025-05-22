# tests/test_api_routes.py
import json
from fastapi.testclient import TestClient

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the LLMOps RAG Pipeline API"}

def test_query_endpoint(client, mock_rag_service):
    # Mock the process_query method
    mock_rag_service.process_query.return_value = {
        "answer": "This is a test answer",
        "sources": [{"content": "Test content", "metadata": {"source": "test"}}],
        "metadata": {"model": "test-model"}
    }
    
    # Test the endpoint
    response = client.post(
        "/api/query",
        json={"question": "What is RAG?"}
    )
    
    assert response.status_code == 200
    assert response.json()["answer"] == "This is a test answer"
    mock_rag_service.process_query.assert_called_once_with(
        "What is RAG?",
        experiment_id=None
    )

def test_query_endpoint_with_experiment(client, mock_rag_service, mock_experiment_service):
    # Mock the process_query method
    mock_rag_service.process_query.return_value = {
        "answer": "This is a test answer",
        "sources": [{"content": "Test content", "metadata": {"source": "test"}}],
        "metadata": {"model": "test-model"}
    }
    
    # Test the endpoint with experiment_id
    response = client.post(
        "/api/query",
        json={"question": "What is RAG?", "experiment_id": "test-experiment"}
    )
    
    assert response.status_code == 200
    assert response.json()["answer"] == "This is a test answer"
    mock_rag_service.process_query.assert_called_once_with(
        "What is RAG?",
        experiment_id="test-experiment"
    )
    mock_experiment_service.log_interaction.assert_called_once()

def test_index_document_endpoint(client, mock_rag_service):
    # Mock the add_document method
    mock_rag_service.add_document.return_value = "doc-123"
    
    # Test the endpoint
    response = client.post(
        "/api/index-document",
        json={
            "content": "This is a test document",
            "metadata": {"source": "test", "author": "tester"}
        }
    )
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "document_id": "doc-123"}
    mock_rag_service.add_document.assert_called_once_with(
        content="This is a test document",
        metadata={"source": "test", "author": "tester"}
    )

def test_create_ab_test_endpoint(client, mock_experiment_service):
    # Mock the create_ab_test method
    mock_experiment_service.create_ab_test.return_value = "test-123"
    
    # Test the endpoint
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
    mock_experiment_service.create_ab_test.assert_called_once_with(
        experiment_id="exp-123",
        variant_a={"template": "Template A", "input_variables": ["query"]},
        variant_b={"template": "Template B", "input_variables": ["query"]},
        evaluation_metric="accuracy"
    )
