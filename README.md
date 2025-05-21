# LLMOps Pipeline for RAG Applications

An end-to-end LLMOps pipeline for fine-tuning, evaluating, and deploying open-source LLMs with Retrieval-Augmented Generation (RAG) capabilities.

## Features

- Fine-tuning pipeline for open-source LLMs using HuggingFace models
- RAG system implementation with LangChain
- FastAPI backend for serving LLM applications
- Version-controlled prompt management with A/B testing
- Experiment tracking with Comet ML
- CI/CD pipeline with GitHub Actions
- Data and model versioning with DVC

## Getting Started

### Prerequisites

- Python 3.9+
- Poetry
- Docker
- Git
- DVC

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shashnavad/llmops-rag-pipeline.git
cd llmops-rag-pipeline
```


2. Install dependencies:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
```


### Running the Application

Start the FastAPI server:

```bash
poetry run uvicorn app.main:app --reload
```


### Running the Pipeline

Execute the DVC pipeline:
```bash
dvc repro
```

## Project Structure

- `app/`: FastAPI application
  - `api/`: API routes
  - `core/`: Core configuration
  - `models/`: Pydantic models
  - `services/`: Business logic services
- `data/`: Data directories (managed by DVC)
  - `raw/`: Raw data
  - `processed/`: Processed data
  - `models/`: Fine-tuned models
  - `prompts/`: Versioned prompts
- `scripts/`: Processing and training scripts
- `tests/`: Test files

## License

This project is licensed under the MIT License - see the LICENSE file for details.
