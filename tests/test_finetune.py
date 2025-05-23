# tests/test_finetune.py
import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.abspath("scripts"))

@pytest.fixture
def mock_dataset():
    with patch("scripts.finetune_model.load_dataset") as mock_load:
        # Create a mock dataset for the train split
        mock_train_dataset = MagicMock()
        mock_train_dataset.map.return_value = mock_train_dataset
        
        # Create the dataset dict
        dataset_dict = {"train": mock_train_dataset}
        mock_load.return_value = dataset_dict
        
        yield mock_train_dataset

def test_finetune(mock_dataset):
    with patch("scripts.finetune_model.AutoModelForCausalLM"), \
         patch("scripts.finetune_model.AutoTokenizer"), \
         patch("scripts.finetune_model.Trainer") as mock_trainer_class, \
         patch("scripts.finetune_model.TrainingArguments"), \
         patch("scripts.finetune_model.DataCollatorForLanguageModeling"), \
         patch("comet_ml.Experiment") as mock_experiment:
        
        # Mock trainer instance
        mock_trainer = MagicMock()
        mock_trainer.evaluate.return_value = {"loss": 1.5, "perplexity": 4.5}
        mock_trainer_class.return_value = mock_trainer
        
        # Mock experiment
        mock_exp = MagicMock()
        mock_experiment.return_value = mock_exp
        
        with patch("builtins.open", mock_open()):
            from scripts.finetune_model import finetune
            finetune()
        
        # Verify training was called
        mock_trainer.train.assert_called_once()
        mock_trainer.save_model.assert_called_once()
