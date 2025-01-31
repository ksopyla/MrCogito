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
from nn.concept_encoder import (
    ConceptEncoderConfig,
    ConceptEncoderForMaskedLM,
    ConceptEncoderWithSimMatrixForMaskedLM
)
import numpy as np
import torch
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_type: str = field(
        default="concept_mlm",
        metadata={"help": "Type of model to train", "choices": ["sim_matrix_mlm", "concept_mlm"]}
    )
    hidden_size: int = field(
        default=768,
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
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "Maximum input sequence length"}
    )

@dataclass
class DataTrainingArguments:
    mlm_probability: float = field(
        default=0.25,
        metadata={"help": "Probability for MLM masking"}
    )
    masking_type: str = field(
        default="concepts",
        metadata={"help": "Masking strategy", "choices": ["random", "whole_word", "concepts"]}
    )
    concept_window: int = field(
        default=3,
        metadata={"help": "Number of neighboring words to mask (concepts)"}
    )
    dataset_percent: int = field(
        default=5,
        metadata={"help": "Percentage of Wikipedia dataset to use"}
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
        default="concept-encoder",
        metadata={"help": "W&B project name"}
    )

def parse_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()

def load_and_preprocess_data(tokenizer, args):
    # Load small subset
    dataset = load_dataset("openwebtext", split=f"train[:{args.dataset_percent}%]")
    
    # May need to install with:
    # pip install git+https://github.com/huggingface/datasets@master
    
    # Rename column to match processing
    dataset = dataset.rename_column("text", "content")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["content"],  # Note different column name
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=True
        )
    
    # Process dataset
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["content", "title", "id"]
    )
    
    return dataset.train_test_split(test_size=0.1)

class NeighborWordMaskCollator(DataCollatorForWholeWordMask):
    """
    This class mask the few nearby whole word, the intuition is that concepts contain multiple nearby words.
    """ 
    def __init__(self, *args, window_size=3, **kwargs):
        super().__init__(*args, **kwargs)
        
        # The window size for masking, defines how many whole words to mask
        self.window_size = window_size

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        This function masks the tokens in the input sequence.
        """
        # First apply whole word masking
        masked_inputs, mask_labels = super().torch_mask_tokens(inputs, special_tokens_mask)
        
        # Expand masks to neighbors
        batch_size, seq_len = inputs.shape
        expanded_mask = torch.zeros_like(masked_inputs, dtype=torch.bool)
        
        for b in range(batch_size):
            # Get original masked positions
            masked_indices = torch.where(mask_labels[b])[0].tolist()
            
            # Expand each mask position
            for idx in masked_indices:
                start = max(0, idx - self.window_size)
                end = min(seq_len, idx + self.window_size + 1)
                expanded_mask[b, start:end] = True
                
        # Apply expanded masking
        random_mask = torch.rand(expanded_mask.shape, device=inputs.device) < self.mlm_probability
        final_mask = expanded_mask & random_mask
        
        # Replace with [MASK] or random token
        masked_inputs = torch.where(
            final_mask,
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token),
            inputs
        )
        
        # Optional: Replace 10% of masked tokens with random words
        random_words = torch.randint(
            len(self.tokenizer), 
            inputs.shape, 
            dtype=torch.long, 
            device=inputs.device
        )
        random_replace = (torch.rand(final_mask.shape, device=inputs.device) < 0.1) & final_mask
        masked_inputs[random_replace] = random_words[random_replace]

        return masked_inputs, final_mask

def main():
    # Parse arguments
    model_args, data_args, training_args = parse_args()
    
    # Update training args with model/data parameters
    training_args.per_device_train_batch_size = 8  # Can be moved to dataclass
    training_args.learning_rate = 1e-4
    training_args.weight_decay = 0.01
    training_args.num_train_epochs = 3
    training_args.seed = 42
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create model config using model_args
    config = ConceptEncoderConfig(
        vocab_size=tokenizer.vocab_size,
        concept_size=model_args.concept_size,
        hidden_size=model_args.hidden_size,
        num_hidden_layers=model_args.num_hidden_layers,
        max_position_embeddings=model_args.max_seq_length,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id
    )
    
    # Load dataset using data_args
    dataset = load_and_preprocess_data(tokenizer, data_args)
    
    # Data collator for dynamic masking
    if data_args.masking_type == "whole_word":
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64
        )
    elif data_args.masking_type == "concepts":
        data_collator = NeighborWordMaskCollator(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64,
            window_size=data_args.concept_window
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64
        )
    
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
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        seed=training_args.seed,
        fp16=training_args.fp_type == "fp16",
        bf16=training_args.fp_type == "bf16",
        gradient_accumulation_steps=2,
        dataloader_num_workers=os.cpu_count(),
        report_to="wandb" if training_args.wandb_project else "none",
        push_to_hub=False,
        remove_unused_columns=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        run_name=f"{model_args.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Add before training
    if training_args.wandb_project:
        import wandb
        wandb.init(
            project=training_args.wandb_project,
            config=vars(training_args),
            name=training_args.run_name,
            sync_tensorboard=True
        )
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
