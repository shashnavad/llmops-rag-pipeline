import os
import json
from typing import Dict, Any
from langchain.prompts import PromptTemplate

class PromptService:
    def __init__(self):
        self.prompts_dir = "data/prompts"
        self.default_template = """
        Answer the question based on the context provided.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:
        """
        
        # Create prompts directory if it doesn't exist
        os.makedirs(self.prompts_dir, exist_ok=True)
        
    def get_prompt_template(self, experiment_id: str = None) -> PromptTemplate:
        """
        Get a prompt template, potentially from an experiment
        """
        if experiment_id:
            # Try to load experiment-specific prompt
            prompt_path = os.path.join(self.prompts_dir, f"{experiment_id}.json")
            if os.path.exists(prompt_path):
                with open(prompt_path, "r") as f:
                    prompt_data = json.load(f)
                return PromptTemplate(
                    template=prompt_data["template"],
                    input_variables=prompt_data["input_variables"]
                )
        
        # Return default prompt
        return PromptTemplate(
            template=self.default_template,
            input_variables=["context", "query"]
        )
    
    def save_prompt_template(self, experiment_id: str, template: str, input_variables: list) -> bool:
        """
        Save a prompt template for an experiment
        """
        prompt_data = {
            "template": template,
            "input_variables": input_variables
        }
        
        prompt_path = os.path.join(self.prompts_dir, f"{experiment_id}.json")
        with open(prompt_path, "w") as f:
            json.dump(prompt_data, f)
            
        return True
