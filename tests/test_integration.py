# tests/test_integration.py
import pytest
from unittest.mock import patch, AsyncMock

@patch("app.services.rag_service.RAGService.process_query")
def test_end_to_end_query(mock_process, client):
    mock_process.return_value = {
        "answer": "Answer to: What is RAG?",
        "sources": [{"content": "Test content", "metadata": {"source": "test"}}],
        "metadata": {"model": "test-model"}
    }
    
    response = client.post(
        "/api/query",
        json={"question": "What is RAG?"}
    )
    
    assert response.status_code == 200
    assert response.json()["answer"] == "Answer to: What is RAG?"

@patch("app.services.rag_service.RAGService.add_document")
def test_end_to_end_document_indexing(mock_add, client):
    mock_add.return_value = "doc-123"
    
    response = client.post(
        "/api/index-document",
        json={
            "content": "This is a test document",
            "metadata": {"source": "test", "author": "tester"}
        }
    )
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "document_id": "doc-123"}
