# tests/test_finetune.py
import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.abspath("scripts"))

from scripts.finetune_model import finetune

@pytest.fixture
def mock_comet():
    with patch("comet_ml.Experiment") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_transformers():
    with patch("scripts.finetune_model.AutoModelForCausalLM") as mock_model, \
         patch("scripts.finetune_model.AutoTokenizer") as mock_tokenizer, \
         patch("scripts.finetune_model.Trainer") as mock_trainer, \
         patch("scripts.finetune_model.TrainingArguments") as mock_args:
        
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_args_instance = MagicMock()
        
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_trainer.return_value = mock_trainer_instance
        mock_args.return_value = mock_args_instance
        
        # Mock evaluate to return metrics
        mock_trainer_instance.evaluate.return_value = {"loss": 1.5, "perplexity": 4.5}
        
        yield {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "trainer": mock_trainer,
            "args": mock_args,
            "trainer_instance": mock_trainer_instance
        }

@pytest.fixture
def mock_dataset():
    with patch("scripts.finetune_model.load_dataset") as mock:
        mock_dataset = MagicMock()
        mock_dataset_dict = {"train": mock_dataset}
        mock_dataset.map.return_value = mock_dataset_dict
        mock.return_value = mock_dataset_dict
        yield mock_dataset

def test_finetune(mock_comet, mock_transformers, mock_dataset):
    # Mock open for writing metrics
    m = mock_open()
    
    with patch("builtins.open", m):
        finetune()
    
    # Check that the model was trained
    mock_transformers["trainer_instance"].train.assert_called_once()
    
    # Check that the model was saved
    mock_transformers["trainer_instance"].save_model.assert_called_once_with("data/models/finetuned")
    mock_transformers["tokenizer"].from_pretrained.return_value.save_pretrained.assert_called_once_with("data/models/finetuned")
    
    # Check that metrics were logged
    mock_comet.log_metrics.assert_called_once_with({"loss": 1.5, "perplexity": 4.5})
    
    # Check that metrics were saved to file
    m.assert_called_once_with("metrics.json", "w")
    handle = m()
    handle.write.assert_called_once_with(json.dumps({"loss": 1.5, "perplexity": 4.5}))
