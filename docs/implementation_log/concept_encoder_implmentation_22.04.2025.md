# Concept Encoder - Implementation log date 2025-04-22 10:00

## What We've Implemented

1. **NeighborWordMaskCollator**
   - Implemented in `training/dataset_preprocess.py`
   - Creates concept-level masking by masking adjacent tokens that form potential concepts
   - Uses a window-based approach to mask neighboring words
   - Respects word boundaries (doesn't break words)
   - Uses higher masking probability (25%) than standard BERT (15%)
   - Includes thorough testing in `tests/test_data_collators.py` with 5 tests that all pass

2. **ConceptEncoderForSequenceClassification**
   - Implemented in `nn/concept_encoder.py`
   - Designed for fine-tuning the pretrained concept encoder on classification tasks
   - Pools concept representations for classification
   - Handles different types of classification problems (single-label, multi-label, regression)
   - Compatible with the GLUE benchmark evaluation

3. **Updated MLM Training**
   - Enhanced `training/mlm_training.py` to properly handle both model types:
     - `ConceptEncoderForMaskedLM` (default)
     - `ConceptEncoderWithSimMatrixForMaskedLM` (alternative approach)
   - Added proper integration with the different masking strategies

4. **GLUE Evaluation**
   - Adapted `training/evaluate_model_on_glue.py` to support both XLNet and ConceptEncoder models
   - Added logic to load pretrained ConceptEncoder models for fine-tuning

## Next Steps

1. **Pretraining the Model**
   - Train the Concept Encoder MLM model using the concept-level masking:
   ```bash
   poetry run python training/mlm_training.py \
     --model_type concept_mlm \
     --masking_type concepts \
     --dataset_name Salesforce/wikitext \
     --dataset_name_subset wikitext-2-raw-v1 \
     --concept_window 2 \
     --num_train_epochs 10 \
     --max_seq_length 256
   ```
   - This will create a checkpoint in `Cache/Training/{timestamp}/{run_name}`

2. **GLUE Evaluation**
   - After pretraining, evaluate the model on GLUE tasks:
   ```bash
   poetry run python training/evaluate_model_on_glue.py \
     --model_type concept \
     --model_name_or_path ./Cache/Training/{timestamp}/{run_name} \
     --tokenizer_name bert-base-uncased \
     --task all \
     --epochs 5 \
     --batch_size 16 \
     --visualize
   ```

3. **Research Extensions**
   - Experiment with different concept sizes and window sizes
   - Compare standard MLM masking with concept-level masking
   - Try both MLM heads (regular concept encoder vs. similarity matrix version)
   - Evaluate on domain-specific datasets where concept understanding is critical

## Architecture Review

The Concept Encoder architecture has the following key components:

1. **Concept Representations**: The model learns a fixed set of concept embeddings that attend to the input text

2. **Cross-Attention**: Concepts attend to input tokens to capture relevant information

3. **Concept Self-Attention**: Concepts attend to each other to refine their representations

4. **Feed-Forward with Gating**: A gated feed-forward network further processes the concept representations

5. **MLM Heads**: Two options for mapping concepts back to tokens:
   - Standard attention-based mapping (`ConceptEncoderForMaskedLM`)
   - Similarity matrix with sparse gating (`ConceptEncoderWithSimMatrixForMaskedLM`)

This architecture should be able to learn and represent abstractions beyond individual tokens, capturing higher-level semantic concepts from the text.

## Troubleshooting

If you encounter issues:

1. **Out-of-Memory Errors**: Reduce batch size, sequence length, or model size
2. **Training Instability**: Lower the learning rate or use gradient clipping
3. **Poor Performance**: Try increasing the number of concepts or training epochs

## Visualization and Analysis

After training, you can analyze what concepts the model has learned by:

1. Finding which input patterns activate specific concepts the most
2. Visualizing the attention patterns between concepts and input tokens
3. Analyzing the similarity between different concepts to find semantic relationships 