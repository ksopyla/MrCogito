"""
https://www.fahdmirza.com/2024/04/fine-tune-phi-3-on-local-custom-dataset.html



## run on multi-gpu
torchrun --standalone --nnodes=1 --nproc-per-node=8 train.py 


"""
# %%


from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from rich import print


from dotenv import dotenv_values

config = dotenv_values(".env")


hf_key = config["HUGGINGFACEHUB_API_TOKEN"]
print(hf_key[0:6])


from huggingface_hub import login
login(token=hf_key)
#%%

model_id = "edbeeching/gpt-neo-125M-imdb"
model_id = "microsoft/Phi-3-mini-4k-instruct"

# EleutherAI/pythia-6.9b
# EleutherAI/pythia-1.4b

MAX_SEQ_LENGTH=2048

# %%
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    # device_map="auto",
    attn_implementation="flash_attention_2",
)


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


# %%


### Datasets
# https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style
#
# https://huggingface.co/datasets/timdettmers/openassistant-guanaco?row=2
#
# chat multi turn: https://huggingface.co/datasets/stingning/ultrachat

# "macadeliccc/opus_samantha" - philosopy, personality, relationships, etc


raw_dataset = load_dataset("macadeliccc/opus_samantha", split="train")

print(raw_dataset[2])
#%%

EOS_TOKEN = tokenizer.eos_token_id


def process_dataset(mydata):
    conversations = mydata["conversations"]

    texts = []
    mapper = {"system": "system\n", "human": "\nuser\n", "gpt": "\nassistant\n"}

    end_mapper = {"system": "", "human": "", "gpt": ""}

    for c in conversations:
        text = "".join(
            f"{mapper[(turn := x['from'])]} {x['value']}\n{end_mapper[turn]}" for x in c
        )

        texts.append(f"{text}{EOS_TOKEN}")

    return {"text": texts}


dataset = raw_dataset.map(process_dataset, batched=True)
# %%
print(dataset["text"][2])

# %%
args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=1,
    output_dir="phi3-samantha",
    optim="paged_adamw_32bit",
    bf16=True,
)


trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    formatting_func=process_dataset,
)


trainer.train()
