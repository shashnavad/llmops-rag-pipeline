import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import comet_ml
from app.core.config import settings

def finetune():
    # Initialize Comet ML experiment
    experiment = comet_ml.Experiment(
        api_key=settings.COMET_API_KEY,
        project_name=settings.COMET_PROJECT_NAME
    )
    experiment.set_name("model-fine-tuning")
    
    # Load model and tokenizer
    model_name = settings.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True
    )
    
    # Load dataset
    dataset = load_dataset("json", data_files="data/processed/training_data.json")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="data/models",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir="logs",
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model("data/models/finetuned")
    tokenizer.save_pretrained("data/models/finetuned")
    
    # Log metrics
    metrics = trainer.evaluate()
    experiment.log_metrics(metrics)
    
    # Save metrics to file
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    
    experiment.end()

if __name__ == "__main__":
    finetune()
