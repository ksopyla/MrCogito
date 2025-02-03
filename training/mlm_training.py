import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    ConceptEncoderWithSimMatrixForMaskedLM
)

from training.dataset_preprocess import load_and_preprocess_text_dataset, NeighborWordMaskCollator


@dataclass
class ModelArguments:
    model_type: str = field(
        default="concept_mlm",
        metadata={"help": "Type of model to train", "choices": ["sim_matrix_mlm", "concept_mlm"]}
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
        metadata={"help": "Masking strategy", "choices": ["random", "whole_word", "concepts"]}
    )
    concept_window: int = field(
        default=3,
        metadata={"help": "Number of neighboring words to mask (concepts)"}
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

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="./outputs",
        metadata={"help": "Output directory for checkpoints"}
    )
    logging_dir: str = field(
        default="./logs",
        metadata={"help": "Logging directory"}
    )
    fp_type: str = field(
        default="fp16",
        metadata={"help": "Floating point precision", "choices": ["fp16", "bf16", "no"]}
    )
    wandb_project: str = field(
        default="MrCogito",
        metadata={"help": "W&B project name"}
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
    
    # initialize the model
    
    # Create model config using model_args
    config = ConceptEncoderConfig(
        vocab_size=tokenizer.vocab_size,
        concept_size=model_args.concept_size,
        hidden_size=model_args.hidden_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=8,
        intermediate_size=model_args.hidden_size * 4,
        max_position_embeddings=max_seq_length,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id
    )
        
    # initialize the model
    model = ConceptEncoderForMaskedLM(config)
    
    
    
    # Data collator for dynamic masking
    if data_args.masking_type == "whole_word":
        # whole word masking - mask the random words (not neighbors)
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64
        )
    elif data_args.masking_type == "concepts":
        # concepts masking - mask the neighboring words
        data_collator = NeighborWordMaskCollator(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64,
            window_size=data_args.concept_window
        )
    else:
        # random masking - the classic one, default
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64
        )
    
    
    # Update training args with model/data parameters
    training_args.per_device_train_batch_size = 8  # Can be moved to dataclass
    training_args.learning_rate = 1e-4
    training_args.weight_decay = 0.01
    training_args.num_train_epochs = 3
    training_args.seed = 42
        
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(training_args.output_dir, datetime.now().strftime("%Y%m%d-%H%M%S")),
        logging_dir=os.path.join(training_args.logging_dir, datetime.now().strftime("%Y%m%d-%H%M%S")),
        overwrite_output_dir=True,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        num_train_epochs=training_args.num_train_epochs,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=training_args.seed,
        fp16=training_args.fp_type == "fp16",
        bf16=training_args.fp_type == "bf16",
        gradient_accumulation_steps=2,
        dataloader_num_workers=2,
        report_to="wandb" if training_args.wandb_project else "none",
        push_to_hub=False,
        remove_unused_columns=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        run_name=f"{model_args.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        use_cpu=False,
        #use_liger_kernel=True
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
    
    # Add before training
    if training_args.wandb_project:
        import wandb
        wandb.init(
            project=training_args.wandb_project,
            config=vars(training_args),
            name=training_args.run_name,
            tensorboard=True,
            sync_tensorboard=True,
            tags=[data_args.masking_type, model_args.model_type, os.environ['COMPUTERNAME']],
            group=f"hostname-{os.environ['COMPUTERNAME']}"
        )
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
