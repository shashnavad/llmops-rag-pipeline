# tests/test_process_data.py
import pytest
from unittest.mock import patch, mock_open, MagicMock
import json
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.abspath("scripts"))

from scripts.process_data import process_data

def test_process_data():
    # Mock data
    raw_data = [
        {"question": "What is RAG?", "answer": "Retrieval-Augmented Generation"},
        {"question": "What is LLMOps?", "answer": "LLM Operations"}
    ]
    
    # Mock os.listdir to return a list of files
    with patch("os.listdir", return_value=["data1.json"]), \
         patch("os.makedirs"), \
         patch("builtins.open", mock_open(read_data=json.dumps(raw_data))) as m:
        
        # Mock file writes
        write_mock = mock_open()
        m.side_effect = [
            m.return_value,  # For reading
            write_mock.return_value  # For writing
        ]
        
        process_data()
    
    # Check that the processed data was written correctly
    write_calls = write_mock.return_value.write.call_args_list
    assert len(write_calls) == 2
    
    # Check first item
    first_item = json.loads(write_calls[0][0][0])
    assert "<|im_start|>user\nWhat is RAG?<|im_end|>" in first_item["text"]
    assert "<|im_start|>assistant\nRetrieval-Augmented Generation<|im_end|>" in first_item["text"]
    
    # Check second item
    second_item = json.loads(write_calls[1][0][0])
    assert "<|im_start|>user\nWhat is LLMOps?<|im_end|>" in second_item["text"]
    assert "<|im_start|>assistant\nLLM Operations<|im_end|>" in second_item["text"]
