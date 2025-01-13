#%%
import os
from transformers import (
    EncoderDecoderModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List
import wandb
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Login to Hugging Face Hub
if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
    login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

# Weights & Biases Configuration - only for metrics logging
os.environ["WANDB_PROJECT"] = "MrCogito"
os.environ["WANDB_WATCH"] = "all"  # Log gradients and parameters
os.environ["WANDB_LOG_MODEL"] = "false"  # Don't save model to wandb

#%%
# Model Configuration
class ModelConfig:
    # Model architecture
    
    # encoders
    # https://huggingface.co/answerdotai/ModernBERT-base
    # 
    encoder_model_name = "answerdotai/ModernBERT-base"
    
    
    #decoders
    # https://huggingface.co/EleutherAI/pythia-70m-deduped
    decoder_model_name = "gpt2" # meta-llama/Llama-3.2-1B
    
    # Training hyperparameters
    batch_size = 8
    learning_rate = 5e-5
    num_train_epochs = 3
    warmup_steps = 500
    weight_decay = 0.01
    
    # Output configuration
    output_dir = "encoder_decoder_model"
    hub_model_id = "ksopyla/concept_enc_dec"
    push_to_hub = True

# Dataset Configuration
class DatasetConfig:
    # Dataset details
    dataset_name = "cnn_dailymail"
    dataset_version = "3.0.0"
    
    # Data processing
    max_input_length = 512
    max_target_length = 128
    train_size = 1000  # Set to None to use full dataset
    eval_size = 100    # Set to None to use full dataset
    
    # Column names in the source dataset
    text_column = "article"
    summary_column = "highlights"
    
    # Names for processed columns
    input_text_column = "input_text"
    target_text_column = "target_text"
    input_ids_column = "input_ids"
    target_ids_column = "labels"  # Using 'labels' as it's expected by the trainer
    attention_mask_column = "attention_mask"
    
    # Cache directory
    cache_dir = "./cache"

model_config = ModelConfig()
dataset_config = DatasetConfig()

def format_dataset(examples):
    """Convert dataset to our expected format with input_text and target_text"""
    return {
        dataset_config.input_text_column: examples[dataset_config.text_column],
        dataset_config.target_text_column: examples[dataset_config.summary_column]
    }

def tokenize_data(examples, encoder_tokenizer, decoder_tokenizer):
    """Tokenize both inputs and targets using their respective tokenizers"""
    # Tokenize inputs with encoder tokenizer
    model_inputs = encoder_tokenizer(
        examples[dataset_config.input_text_column],
        max_length=dataset_config.max_input_length,
        padding="max_length",
        truncation=True,
    )
    
    # Tokenize targets with decoder tokenizer
    target_tokens = decoder_tokenizer(
        examples[dataset_config.target_text_column],
        max_length=dataset_config.max_target_length,
        padding="max_length",
        truncation=True,
    )
    
    # Combine the tokenized data
    model_inputs[dataset_config.target_ids_column] = target_tokens["input_ids"]
    
    return model_inputs

def load_and_prepare_dataset():
    """Load and prepare the dataset using Hugging Face datasets library"""
    print("Loading dataset...")
    dataset = load_dataset(
        dataset_config.dataset_name,
        dataset_config.dataset_version,
        cache_dir=dataset_config.cache_dir
    )
    
    # Take subset if specified
    if dataset_config.train_size is not None:
        train_dataset = dataset["train"].select(range(dataset_config.train_size))
    else:
        train_dataset = dataset["train"]
    
    if dataset_config.eval_size is not None:
        eval_dataset = dataset["validation"].select(range(dataset_config.eval_size))
    else:
        eval_dataset = dataset["validation"]
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Format datasets to have consistent column names
    train_dataset = train_dataset.map(
        format_dataset,
        remove_columns=train_dataset.column_names,
        desc="Formatting train dataset"
    )
    
    eval_dataset = eval_dataset.map(
        format_dataset,
        remove_columns=eval_dataset.column_names,
        desc="Formatting eval dataset"
    )
    
    return train_dataset, eval_dataset

def initialize_model_and_tokenizers():
    """Initialize model and tokenizers"""
    print("Initializing model and tokenizers...")
    # Initialize separate tokenizers for encoder and decoder
    encoder_tokenizer = AutoTokenizer.from_pretrained(model_config.encoder_model_name)
    decoder_tokenizer = AutoTokenizer.from_pretrained(model_config.decoder_model_name)
    
    # Initialize the encoder-decoder model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        model_config.encoder_model_name,
        model_config.decoder_model_name
    )
    
    # Set model configuration
    model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    model.config.eos_token_id = decoder_tokenizer.eos_token_id
    model.config.pad_token_id = decoder_tokenizer.pad_token_id
    model.config.vocab_size = decoder_tokenizer.vocab_size
    
    
    return model, encoder_tokenizer, decoder_tokenizer

def prepare_training_arguments():
    """Prepare training arguments"""
    return TrainingArguments(
        output_dir=model_config.output_dir,
        per_device_train_batch_size=model_config.batch_size,
        learning_rate=model_config.learning_rate,
        num_train_epochs=model_config.num_train_epochs,
        warmup_steps=model_config.warmup_steps,
        weight_decay=model_config.weight_decay,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="wandb" if "WANDB_API_KEY" in os.environ else None,
        push_to_hub=model_config.push_to_hub,
        hub_model_id=model_config.hub_model_id,
        hub_strategy="end",
    )

def main():
    # Initialize model and tokenizers
    model, encoder_tokenizer, decoder_tokenizer = initialize_model_and_tokenizers()
    
    # Load and prepare datasets
    train_dataset, eval_dataset = load_and_prepare_dataset()
    
    # Preprocess datasets with both tokenizers
    print("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_data(x, encoder_tokenizer, decoder_tokenizer),
        batched=True,
        remove_columns=[dataset_config.input_text_column, dataset_config.target_text_column],
        desc="Tokenizing train dataset",
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_data(x, encoder_tokenizer, decoder_tokenizer),
        batched=True,
        remove_columns=[dataset_config.input_text_column, dataset_config.target_text_column],
        desc="Tokenizing eval dataset",
    )
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch")
    eval_dataset.set_format(type="torch")
    
    # Initialize wandb
    if "WANDB_API_KEY" in os.environ:
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            name="concept_enc_dec",
            config={
                "encoder_model": model_config.encoder_model_name,
                "decoder_model": model_config.decoder_model_name,
                "max_input_length": dataset_config.max_input_length,
                "max_target_length": dataset_config.max_target_length,
                "batch_size": model_config.batch_size,
                "learning_rate": model_config.learning_rate,
                "num_train_epochs": model_config.num_train_epochs,
                "warmup_steps": model_config.warmup_steps,
                "weight_decay": model_config.weight_decay,
            },
            tags=["encoder-decoder", "sequence-to-sequence"],
        )
        wandb.watch(model, log="all", log_freq=100)
    
    # Prepare training arguments
    training_args = prepare_training_arguments()
    
    # Initialize trainer with the decoder tokenizer for data collation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=decoder_tokenizer,  # Use decoder tokenizer for generation
        data_collator=DataCollatorForSeq2Seq(decoder_tokenizer, model=model),
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save and push to hub
    if model_config.push_to_hub:
        trainer.push_to_hub(
            commit_message="Training complete",
            tags=["encoder-decoder", "sequence-to-sequence"],
        )
    else:
        model.save_pretrained(model_config.output_dir)
        # Save both tokenizers
        encoder_tokenizer.save_pretrained(os.path.join(model_config.output_dir, "encoder_tokenizer"))
        decoder_tokenizer.save_pretrained(os.path.join(model_config.output_dir, "decoder_tokenizer"))
    
    # Close wandb run
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()

#%%
