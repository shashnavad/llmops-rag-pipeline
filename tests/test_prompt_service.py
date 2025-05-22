# tests/test_prompt_service.py
import pytest
import os
import json
from unittest.mock import patch, mock_open
from app.services.prompt_service import PromptService

@pytest.fixture
def prompt_service():
    with patch("os.makedirs"):
        service = PromptService()
        yield service

def test_get_default_prompt_template(prompt_service):
    # Test getting the default prompt template
    with patch("os.path.exists", return_value=False):
        template = prompt_service.get_prompt_template()
    
    assert "Answer the question based on the context provided" in template.template
    assert "context" in template.input_variables
    assert "query" in template.input_variables

def test_get_experiment_prompt_template(prompt_service):
    # Mock the experiment prompt file
    mock_prompt_data = {
        "template": "Custom template for {{query}}",
        "input_variables": ["query"]
    }
    
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json.dumps(mock_prompt_data))):
        template = prompt_service.get_prompt_template("test-experiment")
    
    assert template.template == "Custom template for {{query}}"
    assert template.input_variables == ["query"]

def test_save_prompt_template(prompt_service):
    # Test saving a prompt template
    m = mock_open()
    with patch("builtins.open", m):
        result = prompt_service.save_prompt_template(
            "test-experiment",
            "Custom template for {{query}}",
            ["query"]
        )
    
    assert result is True
    m.assert_called_once_with(os.path.join("data/prompts", "test-experiment.json"), "w")
    handle = m()
    expected_json = json.dumps({
        "template": "Custom template for {{query}}",
        "input_variables": ["query"]
    })
    handle.write.assert_called_once_with(expected_json)
