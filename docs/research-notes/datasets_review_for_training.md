


# Huggingface pretraining and instruction datasets review



Pretraining datasets:
* bookcorpus
* wikipedia
* allenai/c4
* allenai/olmo-mix-1124
* allenai/dolmino-mix-1124
* arxiv
* EleutherAI/pile
* openwebtext
* dolma by allenai
* dolmino by allenai
* 


Instruction datasets:
* open instruction by allenai - https://allenai.github.io/open-instruct/ 
* tulu 3 sft - https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture 






## Olmo 2 dataset mix



### Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research - 

* HF hub dataset: https://huggingface.co/datasets/allenai/dolma 
* github: https://github.com/allenai/dolma
* paper: https://arxiv.org/pdf/2402.00159 
  
Size: The initial release (v1) contained ~3 trillion tokens, making it one of the largest open datasets available at the time. v1.7 has 2.3T tokens.
Sources (v1.7 mix): Common Crawl (Dolma's processed version & C4), RefinedWeb, StarCoder, Reddit, Semantic Scholar (peS2o), arXiv, StackExchange, Flan Collection, CC News, OpenWebMath, Algebraic Stack, Project Gutenberg, MegaWika, Wikipedia & Wikibooks.
Processing:
Primarily English-only (using fastText for language ID with a permissive threshold).
Quality filtering: Uses heuristics and regular expressions (similar to Gopher/Falcon) to remove ill-formed text, boilerplate, etc., especially from web crawls.
Deduplication: Employs multi-stage deduplication (URL-based for CC, paragraph-based within documents) using Bloom filters.
Risk Mitigation: Detects and masks PII (emails, phone numbers, IPs) using regex. Removes harmful/obscene content using fastText classifiers (with a high threshold to avoid removing informal text).
Decontamination: Removes documents containing paragraphs overlapping with common evaluation datasets (using Bloom filters).
Availability: Openly available on the Hugging Face Hub (allenai/dolma) under the ODC-BY license.
Toolkit: AI2 also released the Dolma toolkit, a high-performance suite for curating large language model datasets, enabling others to replicate or build upon their work.


### Dolmino: an Open Corpus of 1.124 Trillion Tokens for Language Model Pretraining Research - 

* HF hub dataset: https://huggingface.co/datasets/allenai/dolmino-mix-1124 

* paper: https://arxiv.org/pdf/2402.00159



### olmo-mix-1124

* HF hub dataset: https://huggingface.co/datasets/allenai/olmo-mix-1124



## EleutherAI/pile





# Models datasets review


## ModernBERT training datasets review

Tokens: 

General information:


List of mentioned or used datasets: 



## Qwen2 training datasets review

Information takend from Qwen 2 technical report: https://arxiv.org/pdf/2407.10671

Tokens: 7T tokens (7 trillion)

General information: code, mathematics, multilingual 30 languages, hibh-quality data with a lot of data cleaning and filtering

List of mentioned or used datasets: 

## Qwen 2.5 training datasets review

Information takend from Qwen 2.5 technical report: https://arxiv.org/pdf/2412.15115

Tokens: ??

General information: better data filtering than Qwen 2, math, code, synthhetic data

