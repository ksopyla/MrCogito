

# Plan for the next experiments


## Embedding space capabilites

[Reserach analysis on embedding space capabilites](../research-notes/embedding_space_capabilites.md)


## Experiments plans to compare the performance of the weighted, perceiver and multi-query perceiver on the GLUE tasks



Evaluate the performance of the weighted and perceiver encoders on the GLUE tasks.


### Step 1: Concept encoder: Weighted vs perceiver experiments on mini-pile

Goal: Evaluate the performance of the weighted encoders on the mini-pile dataset.The current implementation. Serve as a baseline for the perceiver experiments.


1. Train the perceiver on mini-pile - done 17/01/2026 - conclusions: perceiver is able to learn the concepts and use them in the downstream tasks, but the performance is not as good as the weighted encoder. The 'cls_query' is not able to learn the concepts and use them in the downstream tasks. suggested approaches is to implement the multi-query perceiver Sequence Classification (v2) and the attention pooling for the perceiver Sequence Classification (v3) to see if it can improve the performance.
2. Train the weighted on mini-pile - todo 17/01/2026 - previous implementations was on Wikipedia dataset with the bert-base tokenizer. We should train on mini-pile with the ModernBERT tokenizer - to compare the performance with the perceiver.
3. Evaluate the weighted on the GLUE MRPC task - todo 17/01/2026
4. Evaluate the weighted and perceiver on the all GLUE tasks


### Step 2: Concetp encoders implemeantion changes for ablation studies

Goal: Implement the changes for the concept encoders to be able to test the small embeddings size (32-256) versus the larger concept size (256-4096) and the RoPE positional encoding.

1. Implement for weighted and perceiver encoders change that would allow to have different dims for embeddings and concepts, to be able test the small embeddings size (32-256) versus the larger concept size (256-4096) - todo 
2. Implement the RoPE positional encoding for the weighted and perceiver encoders - todo 
3. Train the weighted and perceiver on the mini-pile dataset to see if the implementation is correct
4. Evaluate the weighted and perceiver on the GLUE MRPC task to make sure if the implementation is correct


### Step 3: Multi-query perceiver Sequence Classification (v2) and attention pooling for the perceiver Sequence Classification (v3)

Goal: Implement new perceiver Sequence Classification heads to see if it can improve the performance in comparison to the weighted encoder.

5.  Implement the multi-query perceiver Sequence Classification (v2)
   ```python
   # Multiple queries instead of one
    self.cls_queries = nn.Parameter(torch.zeros(1, config.num_cls_queries, config.hidden_size))
    # ... cross attention ...
    pooled = decoder_output.mean(dim=1)  # or attention-weighted
   ```
6.  Implement the attention pooling for the perceiver Sequence Classification (v3)
   ```python
        attn_weights = self.concept_attention(concept_repr)  # [B, C, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(concept_repr * attn_weights, dim=1)  # [B, H]
        
        # Option 2: Self-attention + mean pooling
        # refined_concepts, _ = self.concept_self_attn(concept_repr, concept_repr, concept_repr)
        # pooled = refined_concepts.mean(dim=1)
        
        pooled = self.pre_classifier(pooled)
        logits = self.classifier(pooled)
   ```
7.  Evaluate multi-query perceiver Sequence Classification (v2) on the MRPC GLUE task
8.  Compare the results of the weighted, perceiver and multi-query perceiver on the GLUE tasks



## Step 4: Testig the influence of concept losses on performance




### Step 5: Analysis of the concepts


Goal: Analyse and visualize the nature of learned concepts. How specialized are the concepts? How diverse are the concepts? How do the concepts interact with the tokens? How do the concepts interact with each other?

1. Analyze the concepts via: 
    1. [check_model_health.py](../../analysis/check_model_health.py)
    2. [concept_analysis.py](../../analysis/concept_analysis.py)
    3. [concept_analysis_notebook.ipynb](../../analysis/concept_analysis_notebook.ipynb)
2. [concept_analysis_framework.md](../research-notes/concept_analysis_framework.md)