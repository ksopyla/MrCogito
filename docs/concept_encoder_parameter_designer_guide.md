# ConceptEncoder Interactive Parameter Designer

## 🎯 **Purpose**
The interactive parameter designer helps you find optimal ConceptEncoder configurations for specific parameter budgets and constraints.

## 🚀 **Usage**

```bash
poetry run python analysis/concept_encoder_parameter_analysis.py
```

## 📋 **Input Parameters**

### **1. Vocabulary Selection**
Choose from comprehensive vocabulary ranges with real model examples:

| Range | Tokens | Description | Examples |
|-------|--------|-------------|----------|
| **Tiny** | 4K | Rapid prototyping | Custom minimal, Domain-specific |
| **Small** | 8K | Mobile/edge efficient | Specialized domains, Mobile models |
| **Compact** | 16K | Compact but diverse | DistilBERT variants, Efficient models |
| **BERT** | 30K | BERT-family standard | BERT, DeBERTa, ELECTRA |
| **Standard** | 32K | Modern standard | LLaMA 1/2, Mixtral, T5 |
| **Medium** | 50K | Balanced efficiency | GPT-2, RoBERTa, CodeT5 |
| **Large** | 64K | Rich tokenization | GPT-3.5 variants, Claude |
| **XLarge** | 128K | Extra-large modern | LLaMA 3, Qwen 1.5, Mistral |
| **XXLarge** | 256K | Multilingual/code max | Gemma, PaLM 2, Multilingual |
| **Custom** | Any size | Your specification | Custom tokenizers |

### **2. Embedding Dimensions**
- **Token Embedding**: 64, 128, 256, 512, 1024 (power of 2)
- **Concept Embedding**: 128, 256, 512, 1024, 2048 (larger for rich concepts)

### **3. Architecture Parameters**
- **Concept Tokens**: 32, 64, 128, 256 (how many concept tokens)
- **FFN Size**: 1024, 2048, 4096 (feed-forward network width)

### **4. Target Parameters**
- Format: "30M", "100M", "500M", "1B"
- The tool finds configurations within ±10% of target

## 📊 **Output Analysis**

The tool provides:

1. **Multiple optimal configurations** ranked by parameter accuracy
2. **Detailed parameter breakdown** showing where parameters are allocated
3. **CLI command** ready to use with your training script
4. **Implementation notes** for missing features

## 🎯 **Key Insights from Analysis**

### **Tiny Vocabulary (4K tokens)**
```
Tiny 10M Example:
- Token embeddings: 2.6% (0.26M) ✅✅
- Transformer layers: 94.6% (9.4M) ✅✅
- Optimal: 384 hidden, 4 layers, maximum efficiency
```

### **BERT Vocabulary (30K tokens)**
```
BERT 30M Example:
- Token embeddings: 12.8% (3.9M) ✅
- Transformer layers: 85.6% (26.2M) ✅
- Optimal: 512 hidden, 5 layers, good balance
```

### **XLarge Vocabulary (128K tokens)**
```
LLaMA3 30M Example:
- Token embeddings: 27.3% (8.2M) ⚠️
- Transformer layers: 70.7% (21.2M)
- Optimal: 768 hidden, 3 layers, requires tiny token embeddings
```

## 💡 **Optimization Strategies**

### **For Small Parameter Budgets (30M)**
- Use **small vocabularies** (BERT) for max transformer parameters
- **Tiny token embeddings** (64-128 dim) + **large concept embeddings** (512-2048 dim)
- **More layers > wider layers** for concept learning

### **For Medium Parameter Budgets (100M)**
- Can afford larger vocabularies (LLaMA3)
- Still benefit from asymmetric embeddings
- 8+ layers become feasible

### **For Large Parameter Budgets (500M+)**
- Vocabulary size becomes less critical
- Can use full-size embeddings
- Focus on depth (12+ layers)

## 🔧 **Implementation Status**

**✅ Currently Supported:**
- `--hidden_size`
- `--num_hidden_layers` 
- `--concept_size`
- `--tokenizer_name`

**⚠️ Requires Implementation:**
- Separate `token_embedding_dim` and `concept_embedding_dim`
- Variable `intermediate_size` (currently `hidden_size * 4`)

## 📝 **Example Session**

```
Input:
- Vocabulary: BERT (30,522 tokens)
- Token dim: 128, Concept dim: 512
- Concepts: 128, FFN: 2048
- Target: 30M

Output:
--hidden_size 512 \
--num_hidden_layers 5 \
--concept_size 128 \
--tokenizer_name bert-base-uncased
# Note: token_embedding_dim=128, concept_embedding_dim=512
```

## 🎁 **Pro Tips**

1. **Start with BERT vocabulary** for prototyping (fewer parameters in embeddings)
2. **Use 2:1 or 4:1 ratio** for concept:token embedding dimensions
3. **Target 80%+ transformer parameters** for optimal concept learning
4. **Test multiple head counts** - they don't affect parameter count!
5. **Consider power-of-2 dimensions** for GPU efficiency