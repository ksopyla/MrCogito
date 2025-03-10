



## ModernBert evaluations

**Table 1: NLU and Code Benchmarks**

|Model|GLUE|Code|
|---|---|---|
|||CSN|
|**Base**|||
|BERT|84.7|41.2|
|RoBERTa|86.4|44.3|
|DeBERTaV3|88.1|17.5|
|NomicBERT|84.0|41.6|
|GTE-en-MLM|85.6|44.9|
|ModernBERT|88.4|56.4|
|**Large**|||
|BERT|85.2|41.6|
|RoBERTa|88.9|47.3|
|DeBERTaV3|91.4|21.2|
|GTE-en-MLM|87.6|40.5|
|ModernBERT|90.4|59.5|

Key:

- **GLUE:** Average score on the General Language Understanding Evaluation benchmark.
- **CSN:** CodeSearchNet score.
- **SQA:** StackQA score.


This table summarizes the performance of ModernBERT and other models on key downstream tasks related to general understanding, reasoning (as reflected in NLU benchmarks), and code-related tasks


## Llama 3 evaluations
[llama-models/models/llama3_1/eval_details.md at main · meta-llama/llama-models](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md)

This document contains some additional context on the settings and methodology for how we evaluated the Llama 3.1 8B, 70B, and 405B pre-trained and post-trained models.

### Language auto-eval benchmark notes:

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#language-auto-eval-benchmark-notes)

For a given benchmark, we strive to use consistent evaluation settings across all models, including external models. We make every effort to achieve optimal scores for external models, including addressing any model-specific parsing and tokenization requirements. Where the scores are lower for external models than self-reported scores on comparable or more conservative settings, we report the self-reported scores for external models. We are also releasing the data generated as part of evaluations with publicly available benchmarks which can be found on [Llama 3.1 Evals Huggingface collection](https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f). We have also developed a [eval reproduction recipe](https://github.com/meta-llama/llama-recipes/tree/b5f64c0b69d7ff85ec186d964c6c557d55025969/tools/benchmarks/llm_eval_harness/meta_eval_reproduce) that demonstrates how to closely reproduce the Llama 3.1 reported benchmark numbers using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) library and the datasets in [3.1 evals collections](https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f) on selected tasks.

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

For pre-trained models, we use the default few-shot and prompt settings as specified [here](https://github.com/ruixiangcui/AGIEval). The score is averaged over the english subtasks. The maximum generation length is 10 tokens.

### IFEval

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#ifeval)

For post-trained models, we use the default settings as specified [here](https://arxiv.org/pdf/2311.07911). We compute the prompt level scores and instruction level strict and loose accuracy. We then report the average across all the scores.

### HumanEval/HumanEval+

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#humanevalhumaneval)

For both pre-trained and post-trained models, we use a 0-shot config and report pass@1 scores. The maximum generation length is 1024 tokens.

### CommonSenseQA

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#commonsenseqa)

For pre-trained models, we use the same 7-shot config with CoT prompt as in [Wei et al. (2022)](https://arxiv.org/pdf/2201.11903.pdf). We use the MMLU setup for evaluation where we provide all the choices in the prompt and calculate likelihood over choice characters.

### WinoGrande

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#winogrande)

For pre-trained models, we use a choice based setup for evaluation where we fill in the missing blank with the two possible choices and then compute log-likelihood over the suffix. We use a 5-shot config. We use the MMLU setup where we provide all the choices in the prompt and calculate likelihood over choice characters.

### BIG-Bench Hard

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#big-bench-hard)

For pre-trained models, we use a 3-shot config with CoT prompt and compute the average exact match over the subsets in this task. We run this as a generative task. Maximum generation length is 512 tokens.

### SQuAD

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#squad)

For pre-trained models, we use SQuAD v2 with a 1-shot config and report exact match scores. We run this as a generative task. Maximum generation length is 32 tokens. In the prompt, we include the ground truth Q & A pairs for all previous questions pertaining to the same passage. In short, the prompt template takes the form "{few-shot example} {passage} {all previous Q & A pairs for passage} {input question}". For specifics, see the released [evaluation details dataset](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-evals/viewer/Meta-Llama-3.1-8B-evals__squad__details).

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

For both pre-trained and post-trained models, we use the same 8-shot config with CoT prompt as in [Wei et al. (2022)](https://arxiv.org/pdf/2201.11903.pdf) (maj@1). The maximum generation length is 1024 tokens.

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

For pre-trained models, we use the same 4-shot config as in [Lewkowycz et al. (2022)](https://arxiv.org/pdf/2206.14858.pdf) (maj@1). Maximum generation length is 512 tokens.

For post-trained models we use a 0-shot config with Cot prompt. We enhance the exact match using [sympy](https://www.sympy.org/en/index.html) and then use an [equality template](https://github.com/openai/simple-evals/blob/main/common.py#L27-L85) with a judge to resolve complex expressions. Maximum generation length is 5120 tokens. The MATH score represents the full dataset. The scores for MATH-HARD (Lvl 5) are 25.4, 43.8, and 53.4 for the 8B, 70B and 405B models respectively.

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

For the adversarial versions of squad ([Dynabench](https://aclanthology.org/2021.naacl-main.324/) and [Adversarial](https://aclanthology.org/D17-1215/)), we use the same setting as standard SQuAD (1-shot config and exact match as the metric)

### PAWS

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#paws)

For pre-trained models, we use a 5-shot config and report exact match scores. We run this as a generative task. Maximum generation length is 32 tokens.

### GSM Plus

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#gsm-plus)

For pre-trained models, we use the same 8-shot config with CoT prompt as in [Wei et al. (2022)](https://arxiv.org/pdf/2201.11903.pdf) (maj@1). The maximum generation length is 512 tokens.

### Berkeley Function Calling Leaderboard (BFCL)

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#berkeley-function-calling-leaderboard-bfcl)

Benchmark results were achieved by running the open source evaluation repository [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) on commit 7bef000 without any further changes.

### Nexus

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#nexus)

We use the [open-source](https://github.com/nexusflowai/NexusRaven) prompt and evaluation function followed by the [open source notebook](https://github.com/nexusflowai/NexusRaven-V2/blob/master/evaluation_notebook/GPT4_Evaluation/Benchmark_GPT4.ipynb) to compute the scores.

### API Bank

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#api-bank)

We use a 0-shot config with a custom prompt and parsing function to reduce the incidence of false negatives. We also modify the dataset by correcting and completing the ground truth answers that were initially incorrect or incomplete. Second, we improve the evaluation metric to better assess function call correctness by splitting keyword arguments into two groups. We use exact match for keyword arguments that have a unique ground truth, and ROUGE score for those that accept any string with the same semantic meaning as the reference value.

### Gorilla API Bench

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#gorilla-api-bench)

For post-trained models, we use the same 0-shot prompt and evaluation function as proposed in the [original paper](https://arxiv.org/abs/2305.15334). Just like the [open-source](https://github.com/ShishirPatil/gorilla) implementation, we compare the domains of the retrieved API call from the API database with the ground truth. If the domain of the retrieved API is the same as the ground truth and the API exists in the database, it is considered a success. All other scenarios are considered failures.

### TriviaQA-WIKI

[](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md#triviaqa-wiki)

For TrivialQA, we evaluate on the Wiki validation set, use 5-shot config and compute average exact match. We run this as a generative task. Maximum generation length is 24 tokens.



## Qwen 2.5 evaluations
- Dense, easy-to-use, decoder-only language models, available in **0.5B**, **1.5B**, **3B**, **7B**, **14B**, **32B**, and **72B** sizes, and base and instruct variants.
    
- Pretrained on our latest large-scale dataset, encompassing up to **18T** tokens.
    
- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON.
    
- More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.
    
- Context length support up to **128K** tokens and can generate up to **8K** tokens.
    
- Multilingual support for over **29** languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.

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
| :---------------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- |
| MMLU              | 47.5         | 60.9         | 65.6         | 74.2         | 79.7         | 83.3         |
| MMLU-pro          | 15.7         | 28.5         | 34.6         | 45.0         | 51.2         | 55.1         |
| BBH               | 20.3         | 45.1         | 56.3         | 70.4         | 78.2         | 84.5         |
| MATH              | 19.5         | 35.0         | 42.6         | 49.8         | 55.6         | 57.7         |
| HumanEval         | 30.5         | 37.2         | 42.1         | 57.9         | 56.7         | 58.5         |
| MultiPL-E         | 18.9         | 33.1         | 41.2         | 50.3         | 53.5         | 59.4         |
| Multi-Exam        | 30.8         | 47.9         | 54.6         | 59.4         | 70.6         | 75.4         |
| Multi-Understanding | 41.0         | 65.1         | 76.6         | 79.3         | 85.9         | 88.4         |

**Qwen2.5 Instruction-Tuned Model Evaluation (Up to 32B Parameters):**

| Dataset           | Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct | Qwen2.5-32B-Instruct |
| :---------------- | :------------------ | :------------------ | :------------------- | :------------------- |
| MMLU-Pro          | 43.7                | 56.3                | 63.7                 | 69.0                 |
| MATH              | 65.9                | 75.5                | 80.0                 | 83.1                 |
| HumanEval         | 74.4                | 84.8                | 83.5                 | 88.4                 |
| MultiPL-E         | 60.2                | 70.4                | 72.8                 | 75.4                 |
| IFEval            | 58.2                | 71.2                | 81.0                 | 79.5                 |
| Arena-Hard        | N/A                 | 52.0                | 68.3                 | 74.5                 |
| MTbench           | N/A                 | 8.75                | 8.88                 | 9.20                 |

**Multilingual Performance of Instruction-Tuned Models (7B-14B):**

| Dataset               | Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct |
| :--------------------- | :------------------ | :------------------- |
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

