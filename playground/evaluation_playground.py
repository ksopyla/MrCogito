
#%%
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os



#%% Constants for cache directories
DATASET_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))
TOKENIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Tokenizers"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Models"))

def evaluate_model(model_name, task_list, num_fewshot=0):
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=MODEL_DIR
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=TOKENIZER_DIR)
    
    # Make sure tokenizer has necessary tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create HF model wrapper
    hf_model = HFLM(
        pretrained=model_name,
        tokenizer=tokenizer,
        model=model,
        batch_size=8
    )
    
    # Run evaluation
    print(f"Evaluating on tasks: {task_list}")
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=8
    )
    
    # Return results
    return results

# Define tasks to evaluate
tasks_to_evaluate = {
    "mmlu": {"tasks": ["mmlu"], "fewshot": 5},
    "gsm8k": {"tasks": ["gsm8k"], "fewshot": 8},
    "bbh": {"tasks": ["bbh"], "fewshot": 3}
}

# Your model name
model_name = "xlnet-base-cased"

# Run evaluations and collect results
all_results = {}
for benchmark, config in tasks_to_evaluate.items():
    print(f"\nEvaluating {benchmark}...")
    benchmark_results = evaluate_model(
        model_name, 
        config["tasks"], 
        num_fewshot=config["fewshot"]
    )
    all_results[benchmark] = benchmark_results

# Print summary of results
print("\n===== EVALUATION RESULTS =====")
for benchmark, result in all_results.items():
    for task, metrics in result["results"].items():
        print(f"{benchmark} - {task}: {metrics['acc']*100:.2f}%")
