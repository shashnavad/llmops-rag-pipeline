import os
import json
from typing import List, Dict, Any

def process_data():
    """
    Process raw data into a format suitable for fine-tuning
    """
    # Create output directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Read raw data files
    raw_files = os.listdir("data/raw")
    processed_data = []
    
    for file in raw_files:
        if file.endswith(".json"):
            with open(os.path.join("data/raw", file), "r") as f:
                data = json.load(f)
                
                # Process each document
                for item in data:
                    # Format as instruction-response pairs
                    processed_item = {
                        "text": f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n{item['answer']}<|im_end|>"
                    }
                    processed_data.append(processed_item)
    
    # Save processed data
    with open("data/processed/training_data.json", "w") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    process_data()
