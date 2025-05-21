from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any

from app.models.rag import RAGQuery, RAGResponse
from app.services.rag_service import RAGService
from app.services.experiment_service import ExperimentService

router = APIRouter()
rag_service = RAGService()
experiment_service = ExperimentService()

@router.post("/query", response_model=RAGResponse)
async def query_rag(query: RAGQuery):
    """
    Query the RAG system with a user question
    """
    try:
        result = await rag_service.process_query(
            query.question,
            experiment_id=query.experiment_id
        )
        
        # Log the query and response for experiment tracking
        if query.experiment_id:
            experiment_service.log_interaction(
                experiment_id=query.experiment_id,
                query=query.question,
                response=result["answer"],
                retrieved_docs=result["sources"]
            )
            
        return RAGResponse(
            answer=result["answer"],
            sources=result["sources"],
            metadata=result["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index-document")
async def index_document(document: Dict[str, Any]):
    """
    Index a new document in the RAG system
    """
    try:
        doc_id = await rag_service.add_document(
            content=document["content"],
            metadata=document.get("metadata", {})
        )
        return {"status": "success", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{experiment_id}/ab-test")
async def create_ab_test(experiment_id: str, test_config: Dict[str, Any]):
    """
    Create an A/B test for prompt variants
    """
    try:
        test_id = experiment_service.create_ab_test(
            experiment_id=experiment_id,
            variant_a=test_config["variant_a"],
            variant_b=test_config["variant_b"],
            evaluation_metric=test_config["evaluation_metric"]
        )
        return {"status": "success", "test_id": test_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
