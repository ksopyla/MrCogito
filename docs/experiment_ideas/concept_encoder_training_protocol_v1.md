# ConceptEncoder Training Protocol - Initial Baseline
* [🎯 **Experiment Goal**](#-experiment-goal)
* [📏 **MLM pretraining plan**](#-mlm-pretraining-plan)
* [🎭 **Masking Strategies**](#-masking-strategies)
* [📚 **Dataset Strategy \& Storage**](#-dataset-strategy--storage)
* [💰 **Computation Budget \& Multi-GPU Training**](#-computation-budget--multi-gpu-training)
* [❓ **Open Questions**](#-open-questions)
* [🚀 **Proposed Training Sequence**](#-proposed-training-sequence)

## 🎯 **Experiment Goal**

**Primary Objective**: Establish ConceptEncoder baseline and build the experiment pipeline to train the model on the dataset.

**Success Criteria**:
- ConceptEncoder-tiny (100M) achieves >85% F1 on MRPC (beating DistilBERT-66M)
- Training completes without instability (no NaN losses, converges properly)
- Verify that the model is able to learn the concepts and use them in the downstream tasks
- evaluate the model on GLUE tasks, start with MRPC and then move to other tasks


### Hypothesis to verify:

1. Extremely small token embeddings, 4, 16, 32, and larger concept embeddings with fixed concept numbers will be able to learn the semantic relations between the tokens. Intuition is that the small token embeddings will allow for larger vocabularies (embeddings will not take to much memory) and concept embeddings will be a glue that semantically connects the tokens.
    1. we could try small vocab 16k to 128k
    2. we could try small token embeddings 4, 16, 32, 64, 128, 256, 512, 1024, 2048 
    3. we could try larger concept embeddings 128, 256, 512, 1024, 2048, 4096
    4. we could try different number of concepts 64, 128, 256, 512, 1024, 2048, 4096
    5. we could try various FFN sizes 1024, 2048, 4096
2. Are there any differencecs in performance with the same prameter settings? (wide versus deep, is big vocab helps, are token embedings dim matters, is there relationshipp between token and concept embeddings dim?)
3. Larger maksing ratio will transfer the learning singal better, force to learn the concepts. (T5 paper mentions that they experimented with >0.15 probabilities but they didn't find any improvement, revisiting this.)
4. Cross attention allow for better parameter efficiency, smaller model can achive the same performance as larger model.

---

## 📏 **MLM pretraining plan**


### **Model Size Presets**

#### **Micro Model Options (~30-45M params)**


Few experiments configurations to try:

* embeding dim aka hidden_size should be power of 2
* current implementation uses the same embeding size for tokens and concepts, but I want to test different setup: token_embeddings: 4,16,32,64, 128, 256, 512, 1024, and larger concept embeddings 128, 256, 512, 1024, 2048, 4096. The token and concept embeddings are the tupples (4,2048), so small token embeddings will allow for larger vocbularies, and concept embeddings will try to learn more complex relations.
* vocab size ranges from small 16k from diverse corpora, to large 128k (LLama) 256k (Gemma) and other models (get the vocab sizes for other modern models like Qwen, LLama 3 etc.)
* FFN should be power of 2, 1024, 2048, 4096



Based on DistilBERT (66M, 40% of BERT), TinyBERT (14% of BERT), and transformer parameter research:

| Option                 | Params   | Layers | Heads | Concepts | FFN  | Vocab | Max Pos |
|------------------------|----------|--------|-------|----------|------|-------|---------|
| **micro_balanced**     | 37.2M    | 4      | 8     | 64       | 2048 | 30522 | 512     |
| **micro_concept_focused** | 33.7M | 4      | 12    | 128      | 1920 | 30522 | 512     |
| **micro_efficient**    | 30.2M    | 4      | 8     | 56       | 1792 | 30522 | 512     |
| **micro_wide**         | 45.0M    | 3      | 10    | 96       | 2560 | 30522 | 512     |



**Parameter Distribution Analysis:**
- **Embeddings**: ~42-46% (token + position + concept embeddings)
- **Transformer Layers**: ~53-57% (cross-attention + self-attention + gated FFN)
- **Output Head**: ~0.7-0.9% (concept-to-sequence + tied LM head)

**Literature Grounding:**
- All options follow DistilBERT's parameter efficiency principles
- `micro_balanced` uses proven 4-layer depth from DistilBERT
- Gated FFN design inspired by modern efficient architectures
- Concept embeddings add minimal parameter overhead (~1-2%)

#### **Future Model Sizes (To Be Defined)**
| Size      | Target Params | Status |
|-----------|---------------|--------|
| **micro** | ~40M          | TODO   |
| **tiny**  | ~100M         | TODO   |
| **small** | ~500M         | TODO   |
| **base**  | ~1B           | TODO   |
| **large** | ~3B           | TODO   |
| **xlarge**| ~7B           | TODO   |



**Training Priority**: Start with micro → tiny → small, then decide on larger sizes based on results.

---

## 🎭 **Masking Strategies**

### msk1: Whole Word Masking (Baseline)

masking_type = "whole_word"
mlm_probability = 0.15

- **Rationale and intuition**: Standard BERT-style, well-understood baseline. Should work reliably, establishes lower bound

### msk2: whole word masking high probability

masking_type = "whole_word"
mlm_probability = 0.5
- **Rationale and intuition**: Higher probability of masking, more challenging for the model to learn the concepts the Google T5 paper mentions that they experimented with >0.15 probabilities but they didn't find any improvement, revisiting this.

### msk3: Neighbor Word Masking

masking_type = "concepts" 
mlm_probability = 0.15
concept_window = 3  # mask neighboring words

- **Rationale and intuition**: Aligned with concept learning the concept could be understand as neighboring words, so masking the neighboring words could be a good strategy. Better concept representations.

### msk4: Span Masking

masking_type = "span"
mlm_probability = 0.15
span_length_distribution = "poisson(lambda=3)"

- **Rationale and intuition**: Similar to masked diffusion, forces longer-range understanding. If this is different from neighbor masking? I should check in publication how this was implemented.

### msk5: diffusion-based masking

masking_type = "diffusion"
mlm_probability = 0->1 

- **Rationale and intuition**: Similar to masked diffusion as LLaDA paper mentions.

---

## 📚 **Dataset Strategy & Storage**


### **Dataset Size Considerations**

Think carefully about the datasets. 
1. Should I start with a larger one and use a subset? This will give me a better understanding of challenges and limitations, but could be a waste of time if the model is not able to learn the concepts.
2. WikiText-103 is a good starting point, but it is too small. Something a little bit larger would be better, 1GB - 5GB of high quality text.
3. FineWeb-Edu looks like the good option for final training, are there any other high quality datasets in similar size?
4. Is openWeb text as good as fineWeb-Edu?

| Dataset | Tokens | Disk Size | Streaming | Subset Option |
|---------|--------|-----------|-----------|---------------|
| WikiText-103 | 100M | ~500MB | ❌ | Full dataset |
| OpenWebText | 8B | ~40GB | ✅ | 10% subset (4GB) |
| FineWeb-Edu | 1.3T | ~600GB | ✅ | 1% subset (6GB) |
| The Pile | 800B | ~400GB | ✅ | 5% subset (20GB) |

### **Recommended Dataset Plan**

todo

### **Compute devices and their properties**

Compute devices and their properties:

1. Local laptop: 
    - less than 10GB of free disk space, my disk is already full.
    - 64GB of RAM
    - 6 CPU cores
    - 1x RTX 3070
2. Remote deep learning box: 
    - 1TB of SSD disk space
    - 8TB of NAT attached storage
    - 256GB of RAM
    - 4x RTX 3090 (24GB each, 96GB total multi gpu)
    - 64 CPU cores

3. Cloud RunPod:
    - to check, depends on budget :) 



## 💰 **Computation Budget & Multi-GPU Training**

### **Hardware Options**
1. **Local**: 4x RTX 3090 (24GB each, 96GB total)
2. **RunPod**: 4x RTX 4090 (~$2.40/hour) or 4x A100 (~$4.80/hour)

### **Cost Estimation** 
| Model Size | Training Time | Local (Free) | RunPod 4090 | RunPod A100 |
|------------|---------------|--------------|-------------|-------------|
| micro | ~4 hours | ✅ Free | $9.60 | $19.20 |
| tiny | ~8 hours | ✅ Free | $19.20 | $38.40 |
| small | ~24 hours | ✅ Free | $57.60 | $115.20 |
| base | ~48 hours | ✅ Free | $115.20 | $230.40 |
| large | ~96 hours | ⚠️ Long | $230.40 | $460.80 |

**Budget Plan**: Use local 3090s for micro/tiny/small, consider RunPod for base+ if needed.



## ❓ **Open Questions**

### **1. Training Duration & Evaluation**
**Missing**: How many epochs/steps per model? When to stop training?
**Questions**: 
- Should we train to convergence or fixed duration?
- How often to evaluate during training (every N steps)?
- Early stopping criteria?

### **2. Sequence Length Strategy**
**Missing**: Max sequence length affects memory and training time significantly
**Questions**:
- Start with 512 tokens for all sizes
- If the model is able to learn the concepts, we can increase the sequence length.
- Target context window 128K tokens.


### **3. Checkpoint & Resume Strategy**
**Missing**: How to handle long training runs and potential failures
**Questions**:
- Checkpoint frequency (every N steps)?
- How many checkpoints to keep?
- Resume strategy if training fails?

### **4. Learning Rate & Optimization**
**Missing**: While you said defer hyperparameters, LR is crucial for success
**Questions**:
- Use proven schedules (linear warmup + cosine decay)?
- Different LR for different model sizes?
- Minimum LR exploration or use proven values?

---

## 🚀 **Proposed Training Sequence**

### **Week 1: Validation (Local 3090s)**
1. micro + whole_word + WikiText-103 → Quick validation (4 hours)
2. tiny + whole_word + OpenWebText 10% → Serious baseline (8 hours)
3. Evaluate both on MRPC, compare to DistilBERT

### **Week 2: Masking Strategy (Local 3090s)**
1. tiny + neighbor_masking + OpenWebText 10% (8 hours)
2. tiny + span_masking + OpenWebText 10% (8 hours)
3. Compare masking strategies, pick best

### **Week 3: Scaling (Local or RunPod)**
1. small + best_masking + FineWeb 1% (24 hours)
2. Evaluate on MRPC, compare to BERT-base
3. Decide on base/large based on results

---
