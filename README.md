# MrCogito - Concept Encoder Research

A research project developing **Concept Encoder**, a novel transformer architecture that uses concept-level token masking and cross-attention between concept and sequence tokens for improved text understanding and efficient long-context processing.

## 🎯 Research Overview

Traditional transformer models use self-attention between all sequence tokens, which becomes computationally expensive for long contexts (O(n²) complexity). MrCogito explores an alternative approach:

- **Concept Tokens**: A small set of learnable concept embeddings (32-2048 tokens) that capture high-level semantic patterns
- **Cross-Attention**: Instead of expensive sequence-to-sequence self-attention, use cross-attention between concepts and sequence tokens
- **Efficiency**: For 128K context length, concept-based attention requires ~128×2048 operations vs 128K×128K for standard self-attention

### Key Architectural Innovation

The ConceptEncoder uses cross-attention where:
- **Query (Q)**: Concept tokens `[concept_length, embed_dim]` (e.g., 128 concepts)
- **Key/Value (K, V)**: Sequence tokens `[sequence_length, embed_dim]` (e.g., 128K tokens)
- **Result**: Attention matrix `[128 × 128K]` instead of `[128K × 128K]` → **1000x memory reduction**

This architecture is inspired by:
- Memory Transformers (register tokens)
- ConceptBERT (concept-based pre-training)
- Meta's Large Concept Models
- LLaDA (diffusion-based language models)

## 📦 Installation

### Prerequisites

- **Python 3.12+**
- **Poetry** (dependency management)
- **CUDA 12.4** (for GPU training)
- **Windows 11** (primary development environment)

### Setup Instructions

1. **Clone the repository**

```powershell
git clone https://github.com/ksopyla/MrCogito.git
cd MrCogito
```

2. **Install dependencies via Poetry**

```powershell
poetry install
```

This installs all required packages including:
- PyTorch 2.5.0 (with CUDA 12.4 support)
- Transformers 4.47.1
- Datasets, Accelerate, Evaluate
- WandB (for experiment tracking)
- Morfessor (morphological tokenization)

3. **Activate the Poetry environment**

```powershell
poetry shell
```

Your virtual environment will be: `mrcogito-sHhaXiEk-py3.12`

4. **Verify installation**

```powershell
python torch_test.py
```

### CUDA Configuration

The project uses **PyTorch with CUDA 12.4**. The PyPI source is configured in `pyproject.toml`:

```toml
[[tool.poetry.source]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu124"
priority = "explicit"
```

## 🗂️ Project Structure

```
MrCogito/
├── nn/                          # Core model implementation
│   └── concept_encoder.py       # ConceptEncoder architecture
├── training/                    # Training scripts
│   ├── mlm_training.py         # Main MLM pre-training script
│   ├── evaluate_model_on_glue.py  # GLUE benchmark evaluation
│   ├── dataset_preprocess.py   # Data collators and preprocessing
│   └── utils_training.py       # Training utilities
├── scripts/                     # Training/evaluation scripts
│   ├── train_weighted_mlm.ps1  # Windows training script
│   └── train_weighted_mlm_multigpu.sh  # Multi-GPU training
├── tests/                       # Unit tests
│   ├── test_concept_encoder_layer.py
│   └── test_data_collators.py
├── docs/                        # Documentation and research notes
│   ├── research-notes/         # Core research documentation
│   ├── experiments_results/    # Evaluation results
│   └── debugging/              # Troubleshooting guides
├── playground/                  # Prototyping and experimentation
├── Cache/                       # Cached models, datasets, outputs
│   ├── Models/                 # Downloaded HuggingFace models
│   ├── Datasets/               # Cached datasets
│   ├── Evaluation_reports/     # CSV evaluation results
│   └── Training/               # Training checkpoints
└── pyproject.toml              # Poetry dependencies
```

## 🚀 Training

### Quick Start - Single GPU Training

Use the PowerShell script for easy training on Windows:

```powershell
.\scripts\train_weighted_mlm.ps1
```

This trains a **Micro-2 model** (21M parameters) with:
- Model: `weighted_mlm` (simplified weighted combination)
- Hidden size: 256, Layers: 2, Concepts: 128
- Dataset: Wikitext-103
- Batch size: 64, Sequence length: 256
- Learning rate: 5e-4, BF16 precision

### Advanced Training - Custom Configuration

```powershell
python training/mlm_training.py `
    --model_type weighted_mlm `
    --hidden_size 512 `
    --num_hidden_layers 4 `
    --concept_num 256 `
    --mlm_probability 0.15 `
    --max_seq_length 512 `
    --dataset_name "Salesforce/wikitext" `
    --dataset_name_subset "wikitext-103-v1" `
    --per_device_train_batch_size 32 `
    --learning_rate 5e-4 `
    --num_train_epochs 3 `
    --output_dir "./Cache/Training/" `
    --bf16 `
    --report_to "wandb"
```

### Multi-GPU Training (Linux/RunPod)

```bash
bash scripts/train_weighted_mlm_multigpu.sh
```

### Model Variants

The project includes three ConceptEncoder variants:

1. **`concept_mlm`**: Standard ConceptEncoder with cross-attention
2. **`sim_matrix_mlm`**: ConceptEncoder with explicit similarity matrix
3. **`weighted_mlm`**: Simplified weighted combination approach (recommended for initial experiments)

### Model Size Configurations

| Config | Hidden Size | Layers | Concepts | Params |
|--------|-------------|--------|----------|--------|
| Micro-1 | 128 | 2 | 64 | ~8M |
| Micro-2 | 256 | 2 | 128 | ~21M |
| Small | 512 | 4 | 256 | ~85M |
| Base | 768 | 6 | 512 | ~190M |

## 📊 Evaluation

### GLUE Benchmark Evaluation

Evaluate a trained model on GLUE tasks:

```powershell
python training/evaluate_model_on_glue.py `
    --model_type concept `
    --model_path "./Cache/Training/checkpoint-10000" `
    --task mrpc `
    --batch_size 16 `
    --epochs 3
```

### Evaluate on all GLUE tasks:

```powershell
python training/evaluate_model_on_glue.py `
    --model_type concept `
    --model_path "./Cache/Training/checkpoint-10000" `
    --task all `
    --batch_size 32 `
    --epochs 5
```

### Baseline Model Evaluation

Compare against standard encoders (BERT, RoBERTa, etc.):

```powershell
python training/evaluate_model_on_glue.py `
    --model_type bert `
    --task mrpc `
    --batch_size 16 `
    --epochs 3
```

Results are saved to: `Cache/Evaluation_reports/[dataset]-[task]-[model]-[date]-[content].csv`

## 🧪 Testing

Run the test suite:

```powershell
pytest tests/
```

Run specific tests:

```powershell
pytest tests/test_concept_encoder_layer.py
pytest tests/test_data_collators.py
```

## 📈 Monitoring and Logging

The project uses **Weights & Biases (WandB)** for experiment tracking:

- **Project name**: `MrCogito`
- **Run naming format**: `[dataset]-[task]-[model]`
- **Logs location**: `wandb/` folder

Configure WandB:

```powershell
wandb login
```

## 🔬 Research Documentation

Comprehensive research documentation is available in `docs/`:

- **`docs/research-notes/concept_encoder_notes.md`**: Core research ideas and methodology
- **`docs/research-notes/evaluation_strategies.md`**: Evaluation protocols
- **`docs/experiments_results/encoders_glue_evaluation_baseline.md`**: GLUE benchmark results
- **`docs/experiment_ideas/`**: Future research directions

## 🛠️ Development

### Environment Details

- **OS**: Windows 11
- **Shell**: PowerShell
- **Python**: 3.12.6
- **Poetry env**: `mrcogito-sHhaXiEk-py3.12`
- **Env location**: `C:\Users\krzys\AppData\Local\pypoetry\Cache\virtualenvs\mrcogito-sHhaXiEk-py3.12`

### Cache Configuration

The project uses a local `Cache/` directory for all data:

```powershell
$env:HF_HOME = "C:\Users\krzys\Dev Projects\MrCogito\Cache"
$env:HF_DATASETS_CACHE = "C:\Users\krzys\Dev Projects\MrCogito\Cache\Datasets"
```

This is automatically configured in training scripts.

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{mrcogito2024,
  title={MrCogito: Concept Encoder for Efficient Long-Context Processing},
  author={Sopyla, Krzysztof},
  year={2024},
  url={https://github.com/ksopyla/MrCogito}
}
```

## 👤 Author

- 🌐 Website: [ai.ksopyla.com](https://ai.ksopyla.com)
- 📄 Project Page: [ai.ksopyla.com/projects/concept-encoder](https://ai.ksopyla.com/projects/concept-encoder)
- 💼 LinkedIn: [linkedin.com/in/krzysztof-sopyla](https://www.linkedin.com/in/krzysztof-sopyla/)

## 📄 License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please see the research documentation in `docs/` for current research directions and open questions.