# tests/test_experiment_service.py
import pytest
from unittest.mock import patch, MagicMock
from app.services.experiment_service import ExperimentService

@pytest.fixture
def experiment_service():
    service = ExperimentService()
    yield service

def test_create_experiment(experiment_service):
    # Mock comet_ml.Experiment
    mock_experiment = MagicMock()
    mock_experiment.get_key.return_value = "exp-123"
    
    with patch("comet_ml.Experiment", return_value=mock_experiment):
        experiment_id = experiment_service.create_experiment(
            name="Test Experiment",
            tags=["test", "rag"]
        )
    
    assert experiment_id == "exp-123"
    mock_experiment.set_name.assert_called_once_with("Test Experiment")
    mock_experiment.add_tags.assert_called_once_with(["test", "rag"])
    mock_experiment.end.assert_called_once()

def test_create_ab_test(experiment_service):
    # Mock comet_ml.ExistingExperiment
    mock_experiment = MagicMock()
    
    with patch("comet_ml.ExistingExperiment", return_value=mock_experiment), \
         patch("uuid.uuid4", return_value="test-123"):
        test_id = experiment_service.create_ab_test(
            experiment_id="exp-123",
            variant_a={"template": "Template A"},
            variant_b={"template": "Template B"},
            evaluation_metric="accuracy"
        )
    
    assert test_id == "test-123"
    mock_experiment.log_parameter.assert_any_call("ab_test_test-123_variant_a", {"template": "Template A"})
    mock_experiment.log_parameter.assert_any_call("ab_test_test-123_variant_b", {"template": "Template B"})
    mock_experiment.log_parameter.assert_any_call("ab_test_test-123_metric", "accuracy")
    mock_experiment.end.assert_called_once()

def test_log_interaction(experiment_service):
    # Mock comet_ml.ExistingExperiment
    mock_experiment = MagicMock()
    
    with patch("comet_ml.ExistingExperiment", return_value=mock_experiment), \
         patch("uuid.uuid4", return_value="interaction-123"):
        experiment_service.log_interaction(
            experiment_id="exp-123",
            query="What is RAG?",
            response="RAG stands for Retrieval-Augmented Generation",
            retrieved_docs=[{"content": "Test content", "metadata": {"source": "test"}}]
        )
    
    mock_experiment.log_parameter.assert_any_call("interaction_interaction-123_query", "What is RAG?")
    mock_experiment.log_parameter.assert_any_call(
        "interaction_interaction-123_response", 
        "RAG stands for Retrieval-Augmented Generation"
    )
    mock_experiment.log_parameter.assert_any_call("interaction_interaction-123_doc_0", {"content": "Test content", "metadata": {"source": "test"}})
    mock_experiment.end.assert_called_once()
