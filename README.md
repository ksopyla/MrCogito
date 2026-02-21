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

### Model Variants

The project currently explores these architectures:

1. **`weighted_mlm`** (Primary): A simplified approach using weighted combinations of concept tokens. This is currently the most stable variant and recommended for all initial experiments.
2. **`concept_mlm`**: The original ConceptEncoder design with full cross-attention.
3. **`sim_matrix_mlm`**: Variant with explicit similarity matrices.
4. **Encoder-Decoder (Experimental)**: A sequence-to-sequence model (`training/concept_enc_dec.py`) combining ModernBERT (encoder) and GPT-2 (decoder) for abstractive summarization tasks, paving the way for concept-based generation.

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
│   ├── concept_enc_dec.py      # Encoder-Decoder training (Seq2Seq)
│   ├── evaluate_model_on_glue.py  # GLUE benchmark evaluation
│   ├── dataset_preprocess.py   # Data collators and preprocessing
│   └── utils_training.py       # Training utilities
├── scripts/                     # Training/evaluation scripts
│   ├── train_weighted_mlm.ps1      # Windows training script (Single GPU)
│   ├── train_weighted_mlm_multigpu.sh  # Linux Multi-GPU training
│   ├── evaluate_concept_encoder_glue.ps1 # Windows GLUE evaluation
│   └── evaluate_concept_encoder_glue.sh  # Linux GLUE evaluation
├── analysis/                    # Analysis tools
│   └── check_model_health.py    # Model sanity checker
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

### 1. Weighted MLM Training (Windows / Single GPU)

For development and testing on Windows, use the PowerShell script. This trains a "Micro-2" sized model (21M params) on Wikitext-103.

```powershell
.\scripts\train_weighted_mlm.ps1
```

### 2. Weighted MLM Training (Linux / Multi-GPU)

For large-scale training on clusters (e.g., RunPod, Odra), use the bash script which leverages Hugging Face Accelerate for distributed training.

```bash
bash scripts/train_weighted_mlm_multigpu.sh
```

Configuration variables (batch size, GPUs, learning rate) can be edited directly in the script.

### 3. Encoder-Decoder Training (Summarization)

To train the experimental encoder-decoder architecture on CNN/DailyMail:

```powershell
python training/concept_enc_dec.py
```

This script uses `ModelConfig` and `DatasetConfig` classes within the file for easy hyperparameter tuning.

### Model Size Configurations

| Config | Hidden Size | Layers | Concepts | Params |
|--------|-------------|--------|----------|--------|
| Micro-1 | 128 | 2 | 64 | ~8M |
| Micro-2 | 256 | 2 | 128 | ~21M |
| Small | 512 | 4 | 256 | ~85M |
| Base | 768 | 6 | 512 | ~190M |

## 📊 Evaluation

### GLUE Benchmark

We provide unified scripts for evaluating trained models on the GLUE benchmark. These scripts handle formatting, model loading, and result saving automatically.

**Windows (PowerShell):**
```powershell
# Evaluate on MRPC (default)
.\scripts\evaluate_concept_encoder_glue.ps1 -ModelPath "Cache/Training/your_model_folder"

# Evaluate on other tasks
.\scripts\evaluate_concept_encoder_glue.ps1 -ModelPath "..." -Task "sst2"
```

**Linux (Bash):**
```bash
# Usage: script.sh [model_path] [task]
bash scripts/evaluate_concept_encoder_glue.sh "Cache/Training/your_model_folder" "mrpc"
```

Results are saved to: `Cache/Evaluation_reports/[dataset]-[task]-[model]-[date]-[content].csv`

## 🩺 Model Health & Analysis

Before running expensive fine-tuning or evaluations, it is crucial to verify the structural and numerical health of the pre-trained model.

### Health Check Script

The `analysis/check_model_health.py` tool performs comprehensive sanity checks:

*   **Parameter Health**: Scans for NaN/Inf values, extreme magnitudes, and dead neurons (zero variance).
*   **Concept Embeddings**: Checks distribution and collapse (if concepts are too similar).
*   **Forward Pass**: Verifies the model runs without errors and produces diverse logits.
*   **Loss Stability**: Checks if loss computation is stable and positive.

**Usage:**

```powershell
# Basic health check
python analysis/check_model_health.py --model_path "Cache/Training/your_model_folder"

# Detailed weight inspection (per layer statistics)
python analysis/check_model_health.py --model_path "Cache/Training/your_model_folder" --detailed
```

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
- **`docs/3_Evaluations_and_Baselines/canonical_baselines.md`**: GLUE benchmark results
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
