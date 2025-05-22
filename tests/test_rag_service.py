# tests/test_rag_service.py
import pytest
from unittest.mock import MagicMock, patch
import os

from app.services.rag_service import RAGService

@pytest.fixture
def mock_embeddings():
    with patch("app.services.rag_service.HuggingFaceEmbeddings") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_vector_store():
    with patch("app.services.rag_service.Chroma") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_llm():
    with patch("app.services.rag_service.HuggingFacePipeline") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_prompt_service():
    with patch("app.services.rag_service.PromptService") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def rag_service(mock_embeddings, mock_vector_store, mock_llm, mock_prompt_service):
    with patch("app.services.rag_service.AutoTokenizer"), \
         patch("app.services.rag_service.AutoModelForCausalLM"), \
         patch("app.services.rag_service.pipeline"):
        service = RAGService()
        yield service

async def test_process_query(rag_service, mock_vector_store):
    # Mock the retriever
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    
    # Mock the prompt template
    mock_prompt_template = MagicMock()
    rag_service.prompt_service.get_prompt_template.return_value = mock_prompt_template
    
    # Mock the QA chain
    mock_qa_chain = MagicMock()
    mock_qa_chain.return_value = {
        "result": "Test answer",
        "source_documents": [
            MagicMock(page_content="Test content", metadata={"source": "test"})
        ]
    }
    
    with patch("app.services.rag_service.RetrievalQA.from_chain_type", return_value=mock_qa_chain):
        result = await rag_service.process_query("What is RAG?")
    
    assert result["answer"] == "Test answer"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["content"] == "Test content"
    assert result["sources"][0]["metadata"]["source"] == "test"

async def test_add_document(rag_service, mock_vector_store):
    # Mock the add_texts method
    mock_vector_store.add_texts.return_value = ["doc-123"]
    
    doc_id = await rag_service.add_document(
        content="Test content",
        metadata={"source": "test"}
    )
    
    assert doc_id == "doc-123"
    mock_vector_store.add_texts.assert_called_once_with(
        texts=["Test content"],
        metadatas=[{"source": "test"}]
    )
    mock_vector_store.persist.assert_called_once()
