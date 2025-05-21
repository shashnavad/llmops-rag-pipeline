import uuid
from typing import Dict, Any, List
import comet_ml
from datetime import datetime

from app.core.config import settings

class ExperimentService:
    def __init__(self):
        self.comet_api_key = settings.COMET_API_KEY
        self.project_name = settings.COMET_PROJECT_NAME
        
    def create_experiment(self, name: str, tags: List[str] = None) -> str:
        """Create a new experiment for tracking"""
        experiment = comet_ml.Experiment(
            api_key=self.comet_api_key,
            project_name=self.project_name
        )
        
        experiment.set_name(name)
        if tags:
            experiment.add_tags(tags)
            
        experiment_id = experiment.get_key()
        experiment.end()
        
        return experiment_id
    
    def create_ab_test(
        self, 
        experiment_id: str, 
        variant_a: Dict[str, Any], 
        variant_b: Dict[str, Any],
        evaluation_metric: str
    ) -> str:
        """Create an A/B test for prompt variants"""
        test_id = str(uuid.uuid4())
        
        experiment = comet_ml.ExistingExperiment(
            api_key=self.comet_api_key,
            experiment_key=experiment_id
        )
        
        # Log test configuration
        experiment.log_parameter(f"ab_test_{test_id}_variant_a", variant_a)
        experiment.log_parameter(f"ab_test_{test_id}_variant_b", variant_b)
        experiment.log_parameter(f"ab_test_{test_id}_metric", evaluation_metric)
        experiment.log_parameter(f"ab_test_{test_id}_created_at", datetime.now().isoformat())
        
        experiment.end()
        
        return test_id
    
    def log_interaction(
        self, 
        experiment_id: str, 
        query: str, 
        response: str, 
        retrieved_docs: List[Dict[str, Any]]
    ):
        """Log a query-response interaction to the experiment"""
        experiment = comet_ml.ExistingExperiment(
            api_key=self.comet_api_key,
            experiment_key=experiment_id
        )
        
        # Log the interaction
        interaction_id = str(uuid.uuid4())
        experiment.log_parameter(f"interaction_{interaction_id}_query", query)
        experiment.log_parameter(f"interaction_{interaction_id}_response", response)
        experiment.log_parameter(f"interaction_{interaction_id}_timestamp", datetime.now().isoformat())
        
        # Log retrieved documents
        for i, doc in enumerate(retrieved_docs):
            experiment.log_parameter(f"interaction_{interaction_id}_doc_{i}", doc)
            
        experiment.end()
