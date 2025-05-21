import os
from typing import Dict, List, Any
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from app.core.config import settings
from app.services.prompt_service import PromptService

class RAGService:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        
        # Initialize vector store
        self.vector_store = self._init_vector_store()
        
        # Initialize LLM
        self.llm = self._init_llm()
        
        # Initialize prompt service
        self.prompt_service = PromptService()
        
    def _init_vector_store(self):
        # Create directory if it doesn't exist
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        # Initialize ChromaDB
        return Chroma(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=self.embedding_model
        )
    
    def _init_llm(self):
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_NAME,
            device_map="auto",
            load_in_8bit=True
        )
        
        # Create text generation pipeline
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7
        )
        
        # Create LangChain LLM
        return HuggingFacePipeline(pipeline=text_pipeline)
    
    async def process_query(self, question: str, experiment_id: str = None) -> Dict[str, Any]:
        # Get the prompt template (potentially from an experiment)
        prompt_template = self.prompt_service.get_prompt_template(experiment_id)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={
                "prompt": prompt_template
            },
            return_source_documents=True
        )
        
        # Run the chain
        result = qa_chain({"query": question})
        
        # Format the response
        sources = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in result["source_documents"]
        ]
        
        return {
            "answer": result["result"],
            "sources": sources,
            "metadata": {
                "model": settings.MODEL_NAME,
                "prompt_template": prompt_template.template
            }
        }
    
    async def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        # Add document to vector store
        doc_ids = self.vector_store.add_texts(
            texts=[content],
            metadatas=[metadata] if metadata else None
        )
        
        # Persist the vector store
        self.vector_store.persist()
        
        return doc_ids[0]
