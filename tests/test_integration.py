# tests/test_integration.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_rag_pipeline():
    with patch("app.services.rag_service.RAGService.process_query") as mock_process, \
         patch("app.services.rag_service.RAGService.add_document") as mock_add:
        
        # Mock process_query
        async def mock_process_query(question, experiment_id=None):
            return {
                "answer": f"Answer to: {question}",
                "sources": [{"content": "Test content", "metadata": {"source": "test"}}],
                "metadata": {"model": "test-model"}
            }
        
        # Mock add_document
        async def mock_add_document(content, metadata=None):
            return "doc-123"
        
        mock_process.side_effect = mock_process_query
        mock_add.side_effect = mock_add_document
        
        yield

@pytest.mark.usefixtures("mock_rag_pipeline")
def test_end_to_end_query(client):
    # Test the query endpoint
    response = client.post(
        "/api/query",
        json={"question": "What is RAG?"}
    )
    
    assert response.status_code == 200
    assert response.json()["answer"] == "Answer to: What is RAG?"
    assert len(response.json()["sources"]) == 1

@pytest.mark.usefixtures("mock_rag_pipeline")
def test_end_to_end_document_indexing(client):
    # Test the document indexing endpoint
    response = client.post(
        "/api/index-document",
        json={
            "content": "This is a test document",
            "metadata": {"source": "test", "author": "tester"}
        }
    )
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "document_id": "doc-123"}
