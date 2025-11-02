import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import wandb
import argparse
from datetime import datetime
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    set_seed,
    DataCollatorForWholeWordMask,
    HfArgumentParser
)

import numpy as np
import torch
from dataclasses import dataclass, field

from nn.concept_encoder import (
    ConceptEncoderConfig,
    ConceptEncoderForMaskedLM,
    ConceptEncoderWithSimMatrixForMaskedLM,
    ConceptEncoderForMaskedLMWeighted
)

from training.dataset_preprocess import load_and_preprocess_text_dataset
from training.utils_training import (
    print_model_parameters, 
    get_parameter_breakdown
)


@dataclass
class ModelArguments:
    model_type: str = field(
        default="weighted_mlm",
        metadata={"help": "Type of model to train", "choices": ["sim_matrix_mlm", "concept_mlm", "weighted_mlm"]}
    )
    hidden_size: int = field(
        default=256,
        metadata={"help": "Hidden size of the model"}
    )
    num_hidden_layers: int = field(
        default=2,
        metadata={"help": "Number of transformer layers"}
    )
    concept_size: int = field(
        default=128,
        metadata={"help": "Number of concept tokens"}
    )
    

@dataclass
class DataTrainingArguments:
    mlm_probability: float = field(
        default=0.25,
        metadata={"help": "Probability for MLM masking"}
    )
    masking_type: str = field(
        default="random",
        metadata={"help": "Masking strategy", "choices": ["random", "whole_word"]}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum input sequence length"}
    )
    test_size_percent: float = field(
        default=0.1,
        metadata={"help": "Percentage of dataset to use for testing"}
    )
    dataset_name: str = field(
        default="Salesforce/wikitext",
        metadata={"help": "Dataset name to use for training from HuggingFace hub"}
    )
    dataset_name_subset: str | None = field(
        default=None, 
        metadata={"help": "Dataset name subset to use for training from HuggingFace hub, provide if exists"}
    )
    tokenizer_name: str = field(
        default="bert-base-uncased",
        metadata={"help": "Tokenizer name to use for training from HuggingFace hub"}
    )

def parse_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()


def main():
    
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Parse arguments
    model_args, data_args, training_args = parse_args()
    
    # load and preprocess the dataset
    dataset_name = data_args.dataset_name
    dataset_name_subset = data_args.dataset_name_subset
    test_size_percent = data_args.test_size_percent
    tokenizer_name = data_args.tokenizer_name
    max_seq_length = data_args.max_seq_length
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    train_ds, test_ds = load_and_preprocess_text_dataset(tokenizer, 
                                                         dataset_name, 
                                                         dataset_name_subset, 
                                                         "text", 
                                                         test_size_percent = test_size_percent,
                                                         max_seq_length=max_seq_length)
    
    # Create model config using model_args
    # Calculate appropriate number of attention heads based on hidden size
    # Each head should have at least 64 dimensions
    num_attention_heads = max(1, min(8, model_args.hidden_size // 64))
    
    config = ConceptEncoderConfig(
        vocab_size=tokenizer.vocab_size,
        concept_size=model_args.concept_size,
        hidden_size=model_args.hidden_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=model_args.hidden_size * 4,
        max_position_embeddings=max_seq_length,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        tie_word_embeddings=True    
    )
    
    print(f"\nModel Configuration:")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num hidden layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Concept size: {config.concept_size}")
    print(f"Vocab size: {config.vocab_size}")
        
    # initialize the model based on model_type
    print(f"Initializing model of type: {model_args.model_type}")
    if model_args.model_type == "sim_matrix_mlm":
        model = ConceptEncoderWithSimMatrixForMaskedLM(config)
        print("Using ConceptEncoderWithSimMatrixForMaskedLM model")
    elif model_args.model_type == "concept_mlm":
        model = ConceptEncoderForMaskedLM(config)
        print("Using ConceptEncoderForMaskedLM model")
    elif model_args.model_type == "weighted_mlm":
        model = ConceptEncoderForMaskedLMWeighted(config)
        print("Using ConceptEncoderForMaskedLMWeighted model (simplified weighted approach)")
    else:
        raise ValueError(f"Unknown model_type: {model_args.model_type}")
    
    # Print model parameters
    print_model_parameters(model, model_name=f"ConceptEncoder ({model_args.model_type})")
    
    # Optional: Print detailed breakdown
    if training_args.logging_steps <= 10:  # Only show for verbose logging
        breakdown = get_parameter_breakdown(model)
        print("\nParameter breakdown by component:")
        for component, info in breakdown.items():
            if info['params'] > 0:
                print(f"  {component}: {info['params']:,} ({info['params_m']:.2f}M)")
    
    # Data collator for dynamic masking
    if data_args.masking_type == "whole_word":
        # whole word masking - mask the random words (not neighbors)
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64
        )
    else:
        # random masking - the classic one, default
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64
        )
    
    # Set default values if not provided via command line
    if training_args.per_device_train_batch_size is None:
        training_args.per_device_train_batch_size = 24
    if training_args.num_train_epochs is None:
        training_args.num_train_epochs = 10
    if training_args.learning_rate is None:
        training_args.learning_rate = 5e-4
    if training_args.weight_decay is None:
        training_args.weight_decay = 0.01
    if training_args.warmup_steps is None:
        training_args.warmup_steps = 1000
    if training_args.seed is None:
        training_args.seed = 42
        
    # Update training arguments with timestamped directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(training_args.output_dir or "./outputs", f"{model_args.model_type}_{timestamp}")
    logging_dir = os.path.join(training_args.logging_dir or "./logs", f"{model_args.model_type}_{timestamp}_logs")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size or training_args.per_device_train_batch_size,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        num_train_epochs=training_args.num_train_epochs,
        logging_steps=training_args.logging_steps or 100,
        eval_strategy=training_args.eval_strategy or "epoch",
        save_strategy=training_args.save_strategy or "epoch",
        save_safetensors=False,
        seed=training_args.seed,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps or 1,
        dataloader_num_workers=2,
        report_to=["tensorboard", "wandb"],
        push_to_hub=False,
        remove_unused_columns=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        run_name=f"{model_args.model_type}-H{model_args.hidden_size}L{model_args.num_hidden_layers}C{model_args.concept_size}-{timestamp}",
        use_cpu=False,
        warmup_steps=training_args.warmup_steps,
        eval_steps=training_args.eval_steps if training_args.eval_strategy == "steps" else None,
        save_steps=training_args.save_steps if training_args.save_strategy == "steps" else None,
    )
    
    # Initialize wandb
    tags = [
        model_args.model_type,
        "mlm-pretraining",
        data_args.dataset_name,
        data_args.masking_type,
        os.environ['COMPUTERNAME']
    ]
    if data_args.dataset_name_subset:
        tags.append(data_args.dataset_name_subset)

    wandb.init(
        project="MrCogito",
        name=training_args.run_name,
        config=vars(training_args),
        tags=tags,
        group=f"hostname-{os.environ['COMPUTERNAME']}",
        sync_tensorboard=True
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        #tokenizer=tokenizer
    )
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model(
        os.path.join(training_args.output_dir, training_args.run_name)
    )

if __name__ == "__main__":
    main()
