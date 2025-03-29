
* [XLNet](#xlnet)
* [Albert](#albert)
* [ModernBert evaluations](#modernbert-evaluations)
* [XLNet, ModernBert, Albert GLUE results](#xlnet-modernbert-albert-glue-results)
* [Llama 3 evaluations](#llama-3-evaluations)
* [Qwen 2.5 evaluations](#qwen-25-evaluations)
* [LLaDA evaluations](#llada-evaluations)


## XLNet 

Benchmarks results and dataset used for Masked Language Model (MLM) based on XLNet
[XLNet: Generalized Autoregressive Pretraining for Language Understanding arxiv:1906.08237](https://arxiv.org/pdf/1906.08237)

**1. Fair Comparison with BERT**

_Table 1: Fair comparison with BERT. All models are trained using the same data and hyperparameters as in BERT. We use the best of 3 BERT variants for comparison; i.e., the original BERT, BERT with whole word masking, and BERT without next sentence prediction._

| Model                  | SQuAD1.1  | SQuAD2.0  | RACE | MNLI | QNLI | QQP  | RTE  | SST-2 | MRPC | CoLA | STS-B |
|-----------------------|------------|------------|------|------|------|------|------|-------|------|------|-------|
| BERT-Large (Best of 3) | 86.7/92.8 | 82.8/85.5 | 75.1 | 87.3 | 93.0 | 91.4 | 74.0 | 94.0  | 88.7 | 63.7 | 90.2  |
| XLNet-Large-wikibooks  | 88.2/94.0 | 85.1/87.8 | 77.4 | 88.4 | 93.9 | 91.8 | 81.2 | 94.4  | 90.0 | 65.2 | 91.1  |

_Table 2: Comparison with state-of-the-art results on the test set of RACE, a reading comprehension task, and on ClueWeb09-B, a document ranking task. ∗ indicates using ensembles. † indicates our implementations. "Middle" and "High" in RACE are two subsets representing middle and high school difficulty levels. All BERT, RoBERTa, and XLNet results are obtained with a 24-layer architecture with similar model sizes (aka BERT-Large)._

- **RACE and ClueWeb09-B**

|Model|RACE Accuracy||ClueWeb09-B||
|---|---|---|---|---|
||Middle|High|NDCG@20|ERR@20|
|GPT|62.9|57.4|||
|BERT|76.6|70.1|||
|BERT+DCMN|79.5|71.8|||
|RoBERTa|86.5|81.8|||
|XLNet|88.6|84.0|||
|DRMM|||24.3|13.8|
|KNRM|||26.9|14.9|
|Conv|||28.7|18.1|
|BERT|||30.53|18.67|
|XLNet|||31.10|20.28|

_Table 3: Results on SQuAD, a reading comprehension dataset. † marks our runs with the official code. ∗ indicates ensembles. ‡: We are not able to obtain the test results of our latest model on SQuAD1.1 from the organizers after submitting our result for more than one month, and thus report the results of an older version for the SQuAD1.1 test set._

- **SQuAD**

|Model|SQuAD2.0||SQuAD1.1||
|---|---|---|---|---|
||EM|F1|EM|F1|
|**Dev set results (single model)**|||||
|BERT|78.98|81.77|84.1|90.9||
|RoBERTa|86.5|89.4| 88.9|94.6|
|XLNet|87.9|90.6|89.7|95.1|
|**Test set results on leaderboard (single model, as of Dec 14, 2019)**|||||
|BERT|80.005|83.061|85.083|91.835|
|RoBERTa|86.820|89.795|||
|BERT*|||87.433|93.294|
|XLNet|87.926|90.689|89.898|95.080|




**GLUE Benchmark Results**
_Table 5: Results on GLUE. ∗ indicates using ensembles, and † denotes single-task results in a multi-task row. All dev results are the median of 10 runs. The upper section shows direct comparison on dev data and the lower section shows comparison with state-of-the-art results on the public leaderboard._

**Single-task single models on dev set**

| Model   | MNLI      | QNLI | QQP  | RTE  | SST-2 | MRPC | CoLA | STS-B | WNLI |
| ------- | --------- | ---- | ---- | ---- | ----- | ---- | ---- | ----- | ---- |
| BERT-Large | 86.6/-    | 92.3 | 91.3 | 70.4 | 93.2  | 88.0 | 60.6 | 90.0  |  -   |
| RoBERTa-Large | 90.2/90.2 | 94.7 | 92.2 | 86.6 | 96.4  | 90.9 | 68.0 | 92.4  |  -   |
| XLNet   | 90.8/90.8 | 94.9 | 92.3 | 85.9 | 97.0  | 90.8 | 69.0 | 92.5  |  -    |

**Multi-task ensembles on test set**

| Model   | MNLI      | QNLI | QQP  | RTE  | SST-2 | MRPC | CoLA | STS-B | WNLI |
| ------- | --------- | ---- | ---- | ---- | ----- | ---- | ---- | ----- | ---- |
| MT-DNN  | 87.9/87.4 | 96.0 | 89.9 | 86.3 | 96.5  | 92.7 | 68.4 | 91.1  | 89.0 |
| RoBERTa | 90.8/90.2 | 98.9 | 90.2 | 88.2 | 96.7  | 92.3 | 67.8 | 92.2  | 89.0 |
| XLNet   | 90.9/90.9 | 99.0 | 90.4 | 88.5 | 97.1  | 92.9 | 70.2 | 93.0  | 92.5 |




## Albert 

ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
https://arxiv.org/pdf/1909.11942

I have extracted the benchmark results from the "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" paper and formatted them into Markdown tables. Here they are:

**Overall results on the GLUE benchmark**

| Model                     | MNLI-m/mm | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  | Avg. | Params |
| ------------------------- | --------- | ---- | ---- | ----- | ---- | ----- | ---- | ---- | ---- | -------|
| BERT-Base                 | 84.6/84.7 | 71.2 | 90.5 | 93.5  | 52.1 | 89.3  | 88.9 | 70.1 | 80.5 | 110M   |
| BERT-Large                | 86.6/86.9 | 72.1 | 92.7 | 94.9  | 60.5 | 90.0  | 89.4 | 73.2 | 83.5 | 330M   |
| RoBERTa-Large             | 89.0/89.3 | 72.4 | 93.8 | 96.4  | 68.0 | 90.9  | 90.9 | 86.6 | 87.0 | 355M   |
| XLNet-Large               | 89.8/89.9 | 74.1 | 94.7 | 96.5  | 68.5 | 92.0  | 91.8 | 83.8 | 87.3 | 340M   |
| ALBERT-Base               | 84.5/84.5 | 71.2 | 90.8 | 93.3  | 55.8 | 88.6  | 88.2 | 72.3 | 80.6 | 12M    |
| ALBERT-Large              | 86.3/86.6 | 71.8 | 92.2 | 94.7  | 63.6 | 89.4  | 89.1 | 78.7 | 83.3 | 18M    |
| ALBERT-XLarge             | 88.1/88.3 | 73.8 | 93.9 | 96.0  | 67.1 | 90.2  | 90.3 | 83.2 | 86.0 | 59M    |
| ALBERT-XXLarge            | 89.4/89.7 | 74.3 | 94.7 | 96.4  | 68.7 | 91.2  | 91.4 | 86.0 | 87.6 | 235M   |
| ALBERT-XXLarge (Ensemble) | 90.2/90.3 | 74.8 | 95.2 | 96.7  | 71.0 | 91.9  | 92.4 | 88.0 | 88.7 | 235M   |






**Table 2: SQuAD 2.0 results**

| Model                     | EM   | F1   |
| ------------------------- | ---- | ---- |
| BERT-Base                 | 77.4 | 80.8 |
| BERT-Large                | 80.8 | 83.7 |
| XLNet-Large               | 86.0 | 89.4 |
| RoBERTa-Large             | 87.3 | 90.2 |
| ALBERT-Base               | 70.9 | 75.1 |
| ALBERT-Large              | 77.2 | 80.7 |
| ALBERT-XLarge             | 83.4 | 86.5 |
| ALBERT-XXLarge            | 86.4 | 89.4 |
| ALBERT-XXLarge (Ensemble) | 87.8 | 90.8 |

**Table 3: RACE results**

| Model          | Accuracy |
| -------------- | -------- |
| BERT-Base      | 66.9     |
| BERT-Large     | 71.7     |
| XLNet-Large    | 77.3     |
| RoBERTa-Large  | 78.9     |
| ALBERT-Base    | 64.1     |
| ALBERT-Large   | 69.3     |
| ALBERT-XLarge  | 76.5     |
| ALBERT-XXLarge | 79.3     |



## ModernBert evaluations

Benchmarks results and dataset used for Masked Language Model (MLM) based on ModernBert

**Table 1: NLU and Code Benchmarks**

|Model|GLUE|Code CSN|
|---|---|---|
|BERT-Base|84.7|41.2|
|RoBERTa-Base|86.4|44.3|
|DeBERTaV3-Base|88.1|17.5|
|NomicBERT-Base|84.0|41.6|
|GTE-en-MLM-Base|85.6|44.9|
|ModernBERT-Base|88.4|56.4|
|BERT-Large|85.2|41.6|
|RoBERTa-Large|88.9|47.3|
|DeBERTaV3-Large|91.4|21.2|
|GTE-en-MLM-Large|87.6|40.5|
|ModernBERT-Large|90.4|59.5|

Key:

- **GLUE:** Average score on the General Language Understanding Evaluation benchmark.
- **CSN:** CodeSearchNet score.
- **SQA:** StackQA score.


This table summarizes the performance of ModernBERT and other models on key downstream tasks related to general understanding, reasoning (as reflected in NLU benchmarks), and code-related tasks



| Model          | Params | Seq. Len. | CoLA | SST-2 | MRPC | STS-B | QQP  | MNLI | QNLI | RTE  |
| -------------- | ------ | --------- | ---- | ----- | ---- | ----- | ---- | ---- | ---- | ---- |
| **Base**       |        |           |      |       |      |       |      |      |      |      |
| BERT           | 110M   | 512       | 59.0 | 93.1  | 89.5 | 89.4  | 91.4 | 85.4 | 91.6 | 78.2 |
| RoBERTa        | 125M   | 512       | 63.6 | 94.8  | 90.2 | 91.2  | 91.9 | 87.6 | 92.8 | 78.7 |
| DeBERTaV3      | 183M   | 512       | 69.2 | 95.6  | 89.5 | 91.6  | 92.4 | 90.0 | 94.0 | 83.8 |
| MosaicBERT-128 | 137M   | 128       | 58.2 | 93.5  | 89.0 | 90.3  | 92.0 | 85.6 | 91.4 | 83.0 |
| NomicBERT-2048 | 137M   | 2048      | 50.0 | 93.0  | 88.0 | 90.0  | 92.0 | 86.0 | 92.0 | 82.0 |
| GTE-en-MLM     | 137M   | 8192      | 57.0 | 93.4  | 92.1 | 90.2  | 88.8 | 86.7 | 91.9 | 84.8 |
| ModernBERT     | 149M   | 8192      | 65.1 | 96.0  | 92.2 | 91.8  | 92.1 | 89.1 | 93.9 | 87.4 |
| **Large**      |        |           |      |       |      |       |      |      |      |      |
| BERT           | 330M   | 512       | 56.2 | 93.3  | 87.8 | 90.6  | 90.9 | 86.3 | 92.8 | 83.8 |
| RoBERTa        | 355M   | 512       | 68.0 | 96.4  | 90.9 | 92.4  | 92.2 | 90.2 | 94.7 | 86.6 |
| DeBERTaV3      | 434M   | 512       | 75.3 | 96.9  | 92.2 | 93.0  | 93.3 | 91.8 | 96.0 | 92.7 |
| GTE-en-MLM     | 434M   | 8192      | 60.4 | 95.1  | 93.5 | 91.4  | 89.2 | 89.2 | 93.9 | 88.1 |
| ModernBERT     | 395M   | 8192      | 71.4 | 97.1  | 91.7 | 92.8  | 92.7 | 90.8 | 95.2 | 92.1 |



## XLNet, ModernBert, Albert GLUE results







GLUE Benchmark Comparison - By Model Size and Publication
| Model Type | MNLI-m/mm | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B | Avg. |
|------------|-----------|------|-----|-----|-------|------|------|-------|------|
| Base Models |
| BERT-Base (from ALBERT paper) | 84.6/84.7 | 90.5 | 71.2 | 70.1 | 93.5 | 88.9 | 52.1 | 89.3 | 80.5 |
| RoBERTa-Base (from ModernBERT) | - | - | - | - | - | - | - | - | - |
| XLNet-Base | - | - | - | - | - | - | - | - | - |
| ALBERT-Base | 84.5/84.5 | 90.8 | 71.2 | 72.3 | 93.3 | 88.2 | 55.8 | 88.6 | 80.6 |
| ModernBERT-Base | 89.1/89.1 | 93.9 | 92.1 | 87.4 | 96.0 | 92.2 | 65.1 | 91.8 | 88.4 |
| Large Models |
| BERT-Large (from ALBERT paper) | 86.6/86.9 | 92.7 | 72.1 | 73.2 | 94.9 | 89.4 | 60.5 | 90.0 | 83.5 |
| BERT-Large (from XLNet paper) | 86.6/- | 92.3 | 91.3 | 70.4 | 93.2 | 88.0 | 60.6 | 90.0 | - |
| RoBERTa-Large (from XLNet paper) | 90.2/90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4 | - |
| XLNet-Large | 90.8/90.8 | 94.9 | 92.3 | 85.9 | 97.0 | 90.8 | 69.0 | 92.5 | 89.3 |
| ALBERT-Large | 86.3/86.6 | 92.2 | 71.8 | 78.7 | 94.7 | 89.1 | 63.6 | 89.4 | 83.3 |
| ALBERT-XLarge | 88.1/88.3 | 93.9 | 73.8 | 83.2 | 96.0 | 90.3 | 67.1 | 90.2 | 86.0 |
| ALBERT-XXLarge | 89.4/89.7 | 94.7 | 74.3 | 86.0 | 96.4 | 91.4 | 68.7 | 91.2 | 87.6 |
| ModernBERT-Large | 90.8/90.8 | 95.2 | 92.7 | 92.1 | 97.1 | 91.7 | 71.4 | 92.8 | 90.4 |


GLUE Benchmark Comparison - By Model Size and Source
| Model | MNLI-m/mm | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B | Avg. |
|-------|-----------|------|-----|-----|-------|------|------|-------|------|
| **Base Models** |
| BERT-Base (ALBERT paper) | 84.6/84.7 | 90.5 | 71.2 | 70.1 | 93.5 | 88.9 | 52.1 | 89.3 | 80.5 |
| BERT-Base (ModernBERT paper) | 85.4/85.4 | 91.6 | 91.4 | 78.2 | 93.1 | 89.5 | 59.0 | 89.4 | - |
| RoBERTa-Base (ModernBERT paper) | 87.6/87.6 | 92.8 | 91.9 | 78.7 | 94.8 | 90.2 | 63.6 | 91.2 | - |
| ALBERT-Base | 84.5/84.5 | 90.8 | 71.2 | 72.3 | 93.3 | 88.2 | 55.8 | 88.6 | 80.6 |
| ModernBERT-Base | 89.1/89.1 | 93.9 | 92.1 | 87.4 | 96.0 | 92.2 | 65.1 | 91.8 | 88.4 |
| **Large Models** |
| BERT-Large (ALBERT paper) | 86.6/86.9 | 92.7 | 72.1 | 73.2 | 94.9 | 89.4 | 60.5 | 90.0 | 83.5 |
| BERT-Large (XLNet paper) | 86.6/- | 92.3 | 91.3 | 70.4 | 93.2 | 88.0 | 60.6 | 90.0 | - |
| BERT-Large (ModernBERT paper) | 86.3/86.3 | 92.8 | 90.9 | 83.8 | 93.3 | 87.8 | 56.2 | 90.6 | - |
| RoBERTa-Large (XLNet paper) | 90.2/90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4 | - |
| RoBERTa-Large (ALBERT paper) | 89.0/89.3 | 93.8 | 72.4 | 86.6 | 96.4 | 90.9 | 68.0 | 90.9 | 87.0 |
| RoBERTa-Large (ModernBERT paper) | 90.2/90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4 | - |
| XLNet-Large (XLNet paper) | 90.8/90.8 | 94.9 | 92.3 | 85.9 | 97.0 | 90.8 | 69.0 | 92.5 | - |
| XLNet-Large (ALBERT paper) | 89.8/89.9 | 94.7 | 74.1 | 83.8 | 96.5 | 91.8 | 68.5 | 92.0 | 87.3 |
| ALBERT-Large | 86.3/86.6 | 92.2 | 71.8 | 78.7 | 94.7 | 89.1 | 63.6 | 89.4 | 83.3 |
| ALBERT-XLarge | 88.1/88.3 | 93.9 | 73.8 | 83.2 | 96.0 | 90.3 | 67.1 | 90.2 | 86.0 |
| ALBERT-XXLarge | 89.4/89.7 | 94.7 | 74.3 | 86.0 | 96.4 | 91.4 | 68.7 | 91.2 | 87.6 |
| ModernBERT-Large | 90.8/90.8 | 95.2 | 92.7 | 92.1 | 97.1 | 91.7 | 71.4 | 92.8 | 90.4 |


## Llama 3 evaluations
[llama-models/models/llama3_1/eval_details.md at main · meta-llama/llama-models](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md)

This document contains some additional context on the settings and methodology for how we evaluated the Llama 3.1 8B, 70B, and 405B pre-trained and post-trained models.

### Language auto-eval benchmark notes:

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#language-auto-eval-benchmark-notes)

For a given benchmark, we strive to use consistent evaluation settings across all models, including external models. We make every effort to achieve optimal scores for external models, including addressing any model-specific parsing and tokenization requirements. Where the scores are lower for external models than self-reported scores on comparable or more conservative settings, we report the self-reported scores for external models. We are also releasing the data generated as part of evaluations with publicly available benchmarks which can be found on [Llama 3.1 Evals Huggingface collection](https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f). We have also developed a [eval reproduction recipe](https://github.com/meta-llama/llama-recipes/tree/b5f64c0b69d7ff85ec186d964c6c557d55025969/tools/benchmarks/llm_eval_harness/meta_eval_reproduce) that demonstrates how to closely reproduce the Llama 3.1 reported benchmark numbers using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) library and the datasets in [3.1 evals collections](https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f) on selected tasks.

### MMLU

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#mmlu)

For the pre-trained models we use a 5-shot config. To determine the choice character we use the standard MMLU prompt and compare the negative log-likelihood (NLL) of the various choices.

For the post-trained models we report both 5-shot and 0-shot scores. We ask the model to generate the best choice character. The 0-shot scores use a CoT (chain of thought) prompt. The maximum generation lengths for the 5-shot and 0-shot configs are 10 tokens and 1024 tokens respectively.

Macro averages are reported unless otherwise stated. The micro average scores for the various models are: 65.6, 79.0, and 85.4 for the pre-trained 8B, 70B and 405B models respectively for the 5-shot config; 69.44, 84.0, 87.71 for the post-trained 8B, 70B and 405B models respectively for the 5-shot config.

### MMLU-Pro

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#mmlu-pro)

For the pre-trained and post-trained models we use a 5-shot config with CoT prompt. We ask the model to generate the reasoning and the corresponding best choice character. The maximum generation length is 512 tokens for pre-trained setup and 1024 for post-trained setup.

Macro averages are reported unless otherwise stated. The micro average scores for the various models are: 35.6, 52.0, and 59.6 for the pre-trained 8B, 70B and 405B models; 47.0, 65.1, 72.2 for the post-trained 8B, 70B and 405B models.

### ARC-Challenge

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#arc-challenge)

We use the Arc-Challenge subset from the Arc benchmark. For the pre-trained models, we use a 25-shot config and use the MMLU setup for evaluation where we provide all the choices in the prompt and calculate likelihood over choice characters. For the post-trained models, we use 0-shot config and ask the model to generate the choice character. The maximum generation length is 100 tokens.

### GPQA

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#gpqa)

For post-trained models, we use 0-shot config with and without CoT prompt and report exact match scores over the possible options using the main set. Maximum generation length is 96 tokens when not using CoT prompt and 2048 tokens when using the CoT prompt.

### AGIEval English

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#agieval-english)

For pre-trained models, we use the default few-shot and prompt settings as specified [here](https://github.com/ruixiangcui/AGIEval). The score is averaged over the english subtasks. The maximum generation length is 10 tokens.

### IFEval

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#ifeval)

For post-trained models, we use the default settings as specified [here](https://arxiv.org/pdf/2311.07911). We compute the prompt level scores and instruction level strict and loose accuracy. We then report the average across all the scores.

### HumanEval/HumanEval+

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#humanevalhumaneval)

For both pre-trained and post-trained models, we use a 0-shot config and report pass@1 scores. The maximum generation length is 1024 tokens.

### CommonSenseQA

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#commonsenseqa)

For pre-trained models, we use the same 7-shot config with CoT prompt as in [Wei et al. (2022)](https://arxiv.org/pdf/2201.11903.pdf). We use the MMLU setup for evaluation where we provide all the choices in the prompt and calculate likelihood over choice characters.

### WinoGrande

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#winogrande)

For pre-trained models, we use a choice based setup for evaluation where we fill in the missing blank with the two possible choices and then compute log-likelihood over the suffix. We use a 5-shot config. We use the MMLU setup where we provide all the choices in the prompt and calculate likelihood over choice characters.

### BIG-Bench Hard

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#big-bench-hard)

For pre-trained models, we use a 3-shot config with CoT prompt and compute the average exact match over the subsets in this task. We run this as a generative task. Maximum generation length is 512 tokens.

### SQuAD

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#squad)

For pre-trained models, we use SQuAD v2 with a 1-shot config and report exact match scores. We run this as a generative task. Maximum generation length is 32 tokens. In the prompt, we include the ground truth Q & A pairs for all previous questions pertaining to the same passage. In short, the prompt template takes the form "{few-shot example} {passage} {all previous Q & A pairs for passage} {input question}". For specifics, see the released [evaluation details dataset](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-evals/viewer/Meta-Llama-3.1-8B-evals__squad__details).

### QuAC

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#quac)

For pre-trained models, we use a 1-shot config and report the F1 scores. We run this as a generative task. Maximum generation length is 32 tokens.

### BoolQ

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#boolq)

For pre-trained models, we use a 0-shot config and report average accuracy. We run this as a choice task.

### DROP

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#drop)

For pre-trained models, for each validation example, we draw 3 random few-shot examples from the train split and report the F1 scores. The maximum generation length is 32 tokens.

### GSM8K

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#gsm8k)

For both pre-trained and post-trained models, we use the same 8-shot config with CoT prompt as in [Wei et al. (2022)](https://arxiv.org/pdf/2201.11903.pdf) (maj@1). The maximum generation length is 1024 tokens.

### RACE

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#race)

For pre-trained models, we use a 0-shot config. We run this as a choice task. We use the MMLU setup for evaluation where we provide all the choices in the prompt and calculate likelihood over choice characters.

### WorldSense

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#worldsense)

For pre-trained models, we use a 0-shot config. We run this as a choice task. Unlike the original benchmark, we do not normalize the three-option partitions of the benchmark. The chance accuracy is therefore not 0.5, but averages to 0.46.

### MBPP

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#mbpp)

For pre-trained and post-trained models we use a 3-shot config and report pass@1 scores. We run this as a generative task. Maximum generation length is 256 tokens.

### MBPP EvalPlus (base)

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#mbpp-evalplus-base)

For pre-trained and post-trained models we use a 0-shot config and report pass@1 scores. We run this as a generative task. Maximum generation length is 1024 tokens.

### MATH

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#math)

For pre-trained models, we use the same 4-shot config as in [Lewkowycz et al. (2022)](https://arxiv.org/pdf/2206.14858.pdf) (maj@1). Maximum generation length is 512 tokens.

For post-trained models we use a 0-shot config with Cot prompt. We enhance the exact match using [sympy](https://www.sympy.org/en/index.html) and then use an [equality template](https://github.com/openai/simple-evals/blob/main/common.py#L27-L85) with a judge to resolve complex expressions. Maximum generation length is 5120 tokens. The MATH score represents the full dataset. The scores for MATH-HARD (Lvl 5) are 25.4, 43.8, and 53.4 for the 8B, 70B and 405B models respectively.

### SCROLLS

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#scrolls)

For pre-trained models, we use a 5-shot config. Maximum generation length is 32 tokens. Maximum input prompt length is 131072 less the number of tokens generated (i.e. 131040).

### ZeroSCROLLS

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#zeroscrolls)

For post-trained models, we use a 0-shot config. Maximum generation length for QuALITY and SQuALITY is 64 tokens. For Qasper it is 128 tokens. Maximum input prompt length for Llama models is 131072 less the number of tokens generated for each task (i.e. 131008 for QuALITY and SQuALITY and 130944 for Qasper). Maximum input length for non-llama models is 128000 less the number of tokens generated for each task. We ensure that all relevant information is retained in the context for all models for fair comparison.

### InfiniteBench

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#infinitebench)

For post-trained models, we use a 0-shot config. Maximum generation length is 20 for both the En.QA and En.MC tasks and maximum input prompt length is 131052. Maximum input length for non-llama models is 127980. We ensure that all relevant information is retained in the context for all models for fair comparison.

### NIH/Multi-needle

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#nihmulti-needle)

For post-training, we use a 0-shot config. Our context lengths are evenly spaced between 2000 and 131072 in 10 intervals, inclusive of the endpoints for llama models and between 2000 and 128000 for non-llama models. Maximum generation length is 256 tokens.

### Multilingual MGSM

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#multilingual-mgsm)

For post-trained models, we use an 0-shot config with CoT prompt and report exact match (maj@1) scores. Maximum generation length is 2048 tokens. The scores are averaged over all the eleven languages present in the MGSM benchmark, including the ones not supported by Llama models.

### Multilingual MMLU

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#multilingual-mmlu)

For post-trained models, we use a 5-shot config. We run this as a generative task. Maximum generation length is 10 tokens. The scores are individually reported for each and averaged over the seven non-english languages that Llama models support (Portuguese, Spanish, Italian, German, French, Hindi, Thai).

### Multipl-E HumanEval and Multipl-E MBPP

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#multipl-e-humaneval-and-multipl-e-mbpp)

For post-trained models, we use a 0-shot config and report pass@1 scores. Maximum generation length is 512 tokens. Where Multipl-E average is reported, the scores are averaged over all 6 languages in the benchmark.

### PiQA, SiQA, and OpenBookQA

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#piqa-siqa-and-openbookqa)

For pre-trained models, we use a 0-shot config and report average accuracy. We run these as choice task.

### Dynabench SQuAD and Adversarial SQuAD

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#dynabench-squad-and-adversarial-squad)

For the adversarial versions of squad ([Dynabench](https://aclanthology.org/2021.naacl-main.324/) and [Adversarial](https://aclanthology.org/D17-1215/)), we use the same setting as standard SQuAD (1-shot config and exact match as the metric)

### PAWS

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#paws)

For pre-trained models, we use a 5-shot config and report exact match scores. We run this as a generative task. Maximum generation length is 32 tokens.

### GSM Plus

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#gsm-plus)

For pre-trained models, we use the same 8-shot config with CoT prompt as in [Wei et al. (2022)](https://arxiv.org/pdf/2201.11903.pdf) (maj@1). The maximum generation length is 512 tokens.

### Berkeley Function Calling Leaderboard (BFCL)

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#berkeley-function-calling-leaderboard-bfcl)

Benchmark results were achieved by running the open source evaluation repository [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) on commit 7bef000 without any further changes.

### Nexus

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#nexus)

We use the [open-source](https://github.com/nexusflowai/NexusRaven) prompt and evaluation function followed by the [open source notebook](https://github.com/nexusflowai/NexusRaven-V2/blob/master/evaluation_notebook/GPT4_Evaluation/Benchmark_GPT4.ipynb) to compute the scores.

### API Bank

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#api-bank)

We use a 0-shot config with a custom prompt and parsing function to reduce the incidence of false negatives. We also modify the dataset by correcting and completing the ground truth answers that were initially incorrect or incomplete. Second, we improve the evaluation metric to better assess function call correctness by splitting keyword arguments into two groups. We use exact match for keyword arguments that have a unique ground truth, and ROUGE score for those that accept any string with the same semantic meaning as the reference value.

### Gorilla API Bench

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#gorilla-api-bench)

For post-trained models, we use the same 0-shot prompt and evaluation function as proposed in the [original paper](https://arxiv.org/abs/2305.15334). Just like the [open-source](https://github.com/ShishirPatil/gorilla) implementation, we compare the domains of the retrieved API call from the API database with the ground truth. If the domain of the retrieved API is the same as the ground truth and the API exists in the database, it is considered a success. All other scenarios are considered failures.

### TriviaQA-WIKI

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#triviaqa-wiki)

For TrivialQA, we evaluate on the Wiki validation set, use 5-shot config and compute average exact match. We run this as a generative task. Maximum generation length is 24 tokens.



## Qwen 2.5 evaluations
- Dense, easy-to-use, decoder-only language models, available in **0.5B**, **1.5B**, **3B**, **7B**, **14B**, **32B**, and **72B** sizes, and base and instruct variants.
    
- Pretrained on our latest large-scale dataset, encompassing up to **18T** tokens.
    
- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON.
    
- More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.
    
- Context length support up to **128K** tokens and can generate up to **8K** tokens.
    
- Multilingual support for over **29** languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.

Okay, here are the benchmark results for the Qwen2.5 LLM family, focusing on models up to 32B parameters, extracted from the provided documents. I've included descriptions of the dataset acronyms and formatted the results as a Markdown table.

**Dataset Acronyms:**

*   **MMLU:** Massive Multitask Language Understanding
*   **MMLU-Pro:** A more robust and challenging version of MMLU
*   **BBH:** BigBench Hard
*   **ARC-C:** AI2 Reasoning Challenge - Challenge Set
*   **TruthfulQA:** Measures whether a model is truthful in its responses
*   **Winogrande:** An adversarial Winograd Schema Challenge
*   **HellaSwag:**  Evaluates commonsense reasoning for story completion
*   **GPQA:** Graduate-Level Google-Proof Q\&A
*   **TheoremQA:** A theorem-driven question answering dataset
*   **GSM8K:** Grade School Math 8K
*   **MATH:** A dataset of math word problems
*   **HumanEval:**  Evaluates code generation from docstrings
*   **MBPP:** Mostly Basic Python Programming
*   **MultiPL-E:** A scalable and polyglot benchmark for neural code generation
*   **Multi-Exam (M3Exam):** A multilingual exam dataset
*   **IndoMMLU:** MMLU benchmark for Indonesian
*   **ruMMLU:** MMLU benchmark for Russian
*   **mMMLU:** Translated MMLU benchmark
*   **BELEBELE:** A parallel reading comprehension dataset in 122 language variants
*   **XCOPA:** A multilingual dataset for causal commonsense reasoning
*   **XWinograd:** A multilingual Winograd schema challenge
*   **XStoryCloze:** A multilingual story cloze dataset
*   **PAWS-X:** A cross-lingual adversarial dataset for paraphrase identification
*   **MGSM:** Multilingual Grade School Math
*   **Flores-101:** A benchmark for low-resource and multilingual machine translation
*   **LiveCodeBench:** Holistic and contamination free evaluation of large language models for code
*   **IFEval:** Instruction-Following Evaluation
*   **MT-Bench:** A benchmark for evaluating chatbot performance
*   **BLEnD:** A benchmark for LLMs on everyday knowledge in diverse cultures and languages

**Qwen2.5 Base Language Model Evaluation (Up to 32B Parameters):**

| Dataset           | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-3B | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-32B |
|-------------------|--------------|--------------|------------|------------|-------------|-------------|
| MMLU              | 47.5         | 60.9         | 65.6       | 74.2       | 79.7        | 83.3        |
| MMLU-pro          | 15.7         | 28.5         | 34.6       | 45.0       | 51.2        | 55.1        |
| BBH               | 20.3         | 45.1         | 56.3       | 70.4       | 78.2        | 84.5        |
| MATH              | 19.5         | 35.0         | 42.6       | 49.8       | 55.6        | 57.7        |
| HumanEval         | 30.5         | 37.2         | 42.1       | 57.9       | 56.7        | 58.5        |
| MultiPL-E         | 18.9         | 33.1         | 41.2       | 50.3       | 53.5        | 59.4        |
| Multi-Exam        | 30.8         | 47.9         | 54.6       | 59.4       | 70.6        | 75.4        |
| Multi-Understanding | 41.0       | 65.1         | 76.6       | 79.3       | 85.9        | 88.4        |

**Qwen2.5 Instruction-Tuned Model Evaluation (Up to 32B Parameters):**

| Dataset           | Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct | Qwen2.5-32B-Instruct |
|-------------------|---------------------|---------------------|----------------------|----------------------|
| MMLU-Pro          | 43.7                | 56.3                | 63.7                 | 69.0                 |
| MATH              | 65.9                | 75.5                | 80.0                 | 83.1                 |
| HumanEval         | 74.4                | 84.8                | 83.5                 | 88.4                 |
| MultiPL-E         | 60.2                | 70.4                | 72.8                 | 75.4                 |
| IFEval            | 58.2                | 71.2                | 81.0                 | 79.5                 |
| Arena-Hard        | -                   | 52.0                | 68.3                 | 74.5                 |
| MTbench           | -                   | 8.75                | 8.88                 | 9.20                 |

**Multilingual Performance of Instruction-Tuned Models (7B-14B):**

| Dataset               | Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct |
|-----------------------|---------------------|----------------------|
| IFEval (multilingual) | 74.87               | 77.08                |
| AMMLU (Arabic)        | 59.78               | 66.81                |
| JMMLU (Japanese)      | 61.88               | 72.78                |
| KMMLU (Korean)        | 46.59               | 59.71                |
| IndoMMLU (Indonesian) | 56.42               | 65.09                |
| TurkishMMLU (Turkish) | 54.28               | 66.85                |
| MGSM8K (extended)     | 66.11               | 82.27                |
| BLEnD                 | 23.66               | 26.99                |

**Key Takeaways:**

*   The Qwen2.5 series shows significant improvements over previous Qwen models across various tasks and model sizes.
*   The 32B models generally outperform the 14B models, as expected, but the 14B models still offer strong performance.
*   The instruction-tuned models show strong performance on coding and math tasks.
*   Multilingual capabilities are also improved in the Qwen2.5 series.

Please note that some data points were missing for certain models and datasets in the provided text, indicated as "N/A" in the tables. Also, the table is simplified to focus on the most relevant information.


## LLaDA evaluations


They use the [EleutherAI/lm-evaluation-harness: v0.4.3](https://zenodo.org/records/12608602)



Okay, I have extracted the evaluation benchmarks with results from the provided content of the paper. Below is the markdown table summarizing the results:

**Table 1. Benchmark Results of Pre-trained LLMs**

| Model | LLaDA 8B | LLaMA3 8B | LLaMA2 7B | Qwen2 7B | Qwen2.5 7B | Mistral 7B | Deepseek 7B |
|---|---|---|---|---|---|---|---|
| Model Type | Diffusion | AR | AR | AR | AR | AR | AR |
| Training Tokens | 2.3T | 15T | 2T | 7T | 18T | - | 2T |
| **General Tasks** |  |  |  |  |  |  |  |
| MMLU (5-shot) | 65.9 | 65.4 | 45.9 | 70.3 | 74.2 | 64.2 | 48.2 |
| BBH (3-shot) | 49.8 | 57.6 | 37.3 | 62.3 | 70.4 | 56.1 | 39.5 |
| ARC-C (0-shot) | 47.9 | 53.1 | 46.3 | 60.6 | 63.7 | 60.0 | 48.1 |
| Hellaswag (0-shot) | 72.5 | 79.1 | 76.0 | 80.7 | 80.2 | 83.3 | 75.4 |
| TruthfulQA (0-shot) | 46.4 | 44.0 | 39.0 | 54.2 | 56.4 | 42.2 | - |
| WinoGrande (5-shot) | 74.8 | 77.3 | 72.5 | 77.0 | 75.9 | 78.4 | 70.5 |
| PIQA (0-shot) | 74.4 | 80.6 | 79.1 | - | - | - | 79.2 |
| **Mathematics & Science** |  |  |  |  |  |  |  |
| GSM8K (4-shot) | 70.7 | 53.1 | 14.3 | 80.2 | 85.4 | 36.2 | 17.4 |
| Math (4-shot) | 27.3 | 15.1 | 3.2 | 43.5 | 49.8 | 10.2 | 6.0 |
| GPQA (5-shot) | 26.1 | 25.9 | 25.7 | 30.8 | 36.4 | 24.7 | - |
| **Code** |  |  |  |  |  |  |  |
| HumanEval (0-shot) | 33.5 | 34.2 | 12.8 | 51.2 | 57.9 | 29.3 | 26.2 |
| HumanEval-FIM (2-shot) | 73.8 | 73.3 | 26.9 | - | - | - | - |
| MBPP (4-shot) | 38.2 | 47.4 | 18.4 | 64.2 | 74.9 | 51.1 | 39.0 |
| **Chinese** |  |  |  |  |  |  |  |
| CMMLU (5-shot) | 69.9 | 50.7 | 32.5 | 83.9 | - | - | 47.2 |
| C-Eval (5-shot) | 70.5 | 51.7 | 34.0 | 83.2 | - | - | 45.0 |

**Table 2. Benchmark Results of Post-trained LLMs**

| Model | LLaDA 8B | LLaMA3 8B | LLaMA2 7B | Qwen2 7B | Qwen2.5 7B | Gemma2 9B | Deepseek 7B |
|---|---|---|---|---|---|---|---|
| Model Type | Diffusion | AR | AR | AR | AR | AR | AR |
| Training Tokens | 2.3T | 15T | 2T | 7T | 18T | 8T | 2T |
| Post-training | SFT | SFT+RL | SFT+RL | SFT+RL | SFT+RL | SFT+RL | SFT+RL |
| Alignment pairs | 4.5M | - | - | 0.5M + | 1M + | 0.15M - | 1.5M + |
| **General Tasks** |  |  |  |  |  |  |  |
| MMLU (5) | 65.5 | 68.4 | 44.1 | - | - | - | 49.4 |
| MMLU-pro (0) | 37.0 | 41.9 | 4.6 | 44.1 | 56.3 | 52.1 | - |
| Hellaswag (0) | 74.6 | 75.5 | 51.5 | - | - | - | 68.5 |
| ARC-C (0) | 88.5 | 82.4 | 57.3 | - | - | - | 49.4 |
| **Mathematics & Science** |  |  |  |  |  |  |  |
| GSM8K (4) | 78.6 | 78.3 | 29.0 | 85.7 | 91.6 | 76.7 | 63.0 |
| Math (0) | 26.6 | 29.6 | 3.8 | 52.9 | 75.5 | 44.3 | 15.8 |
| GPQA (5) | 31.8 | 31.9 | 28.4 | 34.3 | 36.4 | 32.8 | - |
| **Code** |  |  |  |  |  |  |  |
| HumanEval (0) | 47.6 | 59.8 | 16.5 | 79.9 | 84.8 | 68.9 | 48.2 |
| MBPP (4) | 34.2 | 57.6 | 20.6 | 67.2 | 79.2 | 74.9 | 35.2 |

