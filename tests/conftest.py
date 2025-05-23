# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_rag_service():
    with patch("app.api.routes.rag_service") as mock_service:
        yield mock_service

@pytest.fixture
def mock_experiment_service():
    with patch("app.api.routes.experiment_service") as mock_service:
        yield mock_service
