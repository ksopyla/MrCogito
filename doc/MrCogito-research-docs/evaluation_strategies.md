# Evaluation stratedy



## Evaluation for Masked Language Model based on Concept Encoder


For a masked language model with concept-based encoding, I recommend the following evaluation approach:

### Core MLM Tasks:

* GLUE Benchmark: Tests general language understanding, includes tasks like sentiment analysis, paraphrase detection, and natural language inference, particularly suitable for evaluating concept understanding
    * Fine-tune the model on GLUE tasks and evaluate the performance based on ModernBert example **[finetune_modernbert_on_glue.ipynb](https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/finetune_modernbert_on_glue.ipynb)**
* SQuAD (Stanford Question Answering Dataset): Tests reading comprehension and information extraction, Good for evaluating how well your concept encoder captures and represents information
* CodeSearchNet (CSN): Evaluates code understanding capabilities, Important since your model uses cross-attention which could be particularly effective for code
* Concept-Specific Tasks:
    * Entity Typing: Tests how well the model understands and categorizes entities, Aligns with your concept-based approach
    * Conceptual Knowledge Probing: Evaluates how well the model captures abstract concepts, Can use datasets like ConceptNet or WordNet
* Long-Context Understanding:
    * Long Document Classification: Test how well your model handles longer contexts, Important given your focus on concept-level understanding
    * Document-Level Question Answering: Evaluate ability to understand and reason across longer documents

#### Details of Benchmarks

1. GLUE (General Language Understanding Evaluation)
A collection of diverse natural language understanding tasks designed to evaluate model performance across different linguistic challenges.
Subtasks:
    1. **MNLI** (Multi-Genre Natural Language Inference): 393k train/20k dev examples. Determine if a premise entails, contradicts, or is neutral to a hypothesis. Metric: Accuracy (matched/mismatched sets).
    2. **QQP** (Quora Question Pairs): 364k train/40k dev examples. Determine if two questions are semantically equivalent. Metrics: Accuracy and F1.
    3. **QNLI** (Question Natural Language Inference): 105k train/5.5k dev examples. Determine if a sentence contains the answer to a question. Metric: Accuracy.
    4. **SST-2** (Stanford Sentiment Treebank): 67k train/872 dev examples. Binary sentiment classification of movie reviews. Metric: Accuracy.
    5. **CoLA** (Corpus of Linguistic Acceptability): 8.5k train/1k dev examples. Determine if a sentence is grammatically acceptable. Metric: Matthews correlation.
    6. **STS-B** (Semantic Textual Similarity Benchmark): 7k train/1.5k dev examples. Rate similarity of sentence pairs on a 1-5 scale. Metric: Pearson/Spearman correlation.
    7. **MRPC** (Microsoft Research Paraphrase Corpus): 3.7k train/408 dev examples. Identify if sentence pairs are paraphrases. Metrics: Accuracy and F1.
    8. **RTE** (Recognizing Textual Entailment): 2.5k train/277 dev examples. Determine if one text entails another. Metric: Accuracy.
    9. **WNLI** (Winograd Natural Language Inference): 634 train/71 dev examples. Resolve pronoun references. Metric: Accuracy.
2. Reading Comprehension Tasks
    1. **SQuAD** (Stanford Question Answering Dataset)
        * SQuAD 1.1: 87k train/10k dev/10k test examples. Find answer spans in passages for questions. Metrics: Exact Match (EM) and F1.
        * SQuAD 2.0: 130k train/12k dev/9k test examples. Extension that includes unanswerable questions. Metrics: EM and F1.
    2. **RACE** (ReAding Comprehension from Examinations)
        * 28k passages with 100k questions from English exams for Chinese students. Multiple-choice reading comprehension divided into Middle and High school difficulty. Metric: Accuracy.
3. Code Understanding Tasks
    * **CodeSearchNet** (CSN)
        * Evaluates models' ability to understand and search code across multiple programming languages. Metrics: Mean Reciprocal Rank (MRR).
    * **HumanEval**
        * A dataset of 164k single-step problems drawn from real GitHub code. Metric: Accuracy.









## Evaluation for Seq2Seq (Encoder-Decoder) based on Concept Encoder

ToDo: extend the list in the future, this is just a draft

These benchmarks are primarily designed for generative language models and may not be the best fit for your MLM based Concept Encoder. 


* MMLU (Massive Multitask Language Understanding): A benchmark that tests models across 57 subjects including STEM, humanities, social sciences, and more. It's designed to measure a model's ability to understand and reason across diverse domains.
* GSM8K (Grade School Math Word Problems): A dataset of 8,500 grade school math word problems that tests a model's ability to solve mathematical reasoning problems.
* BBH (Big Bench Hard): A subset of the BIG-bench benchmark that focuses on particularly challenging reasoning tasks that are difficult for current language models.
* todo extend the list in the future





## Evaluation for diffusion generation based on Concept Encoder

ToDo: for future
