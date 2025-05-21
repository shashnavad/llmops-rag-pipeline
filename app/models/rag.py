from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class RAGQuery(BaseModel):
    question: str
    experiment_id: Optional[str] = None

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
