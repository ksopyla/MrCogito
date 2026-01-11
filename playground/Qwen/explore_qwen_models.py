#%%
"""
Qwen Models Interactive Exploration Guide
=========================================
An interactive guide to explore Qwen family models step by step.
Each cell focuses on a specific aspect for learning and experimentation.

Run cells one by one, modify parameters, and explore!
"""

import os
import torch
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import numpy as np

console = Console()

#%%
# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================
MODEL_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Models"))
print(f"[green]Model cache directory:[/green] {MODEL_CACHE_DIR}")

# Available Qwen models to explore
QWEN_MODELS = {
    "qwen2_5_omni_thinker": {
        "name": "Qwen/Qwen2.5-Omni-7B",
        "description": "Qwen2.5 Omni Thinker (text-only, lighter)",
        "class": Qwen2_5OmniThinkerForConditionalGeneration,
    },
    "qwen2_5_omni_full": {
        "name": "Qwen/Qwen2.5-Omni-7B",
        "description": "Qwen2.5 Omni Full (with audio generation)",
        "class": Qwen2_5OmniForConditionalGeneration,
    },
    "qwen2_audio": {
        "name": "Qwen/Qwen2-Audio-7B-Instruct",
        "description": "Qwen2 Audio (audio-focused)",
        "class": AutoModelForSeq2SeqLM,
    },
}

print("\n[bold cyan]Available Models:[/bold cyan]")
for key, info in QWEN_MODELS.items():
    print(f"  {key}: {info['description']}")

#%%
# ============================================================================
# CELL 1: Load Qwen2.5-Omni Thinker Model
# ============================================================================
# This cell loads the Thinker-only version (lighter, faster)
# The Thinker handles text generation and reasoning

model_name = "Qwen/Qwen2.5-Omni-7B"
print(f"\n[bold yellow]Loading: {model_name} (Thinker-only)[/bold yellow]")

# Load processor
processor = Qwen2_5OmniProcessor.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_DIR
)
print("[green]✓ Processor loaded[/green]")

# Load Thinker-only model (lighter, no audio generation)
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
print("[green]✓ Model loaded[/green]")

# Explore: Check what device the model is on
print(f"\n[cyan]Model device:[/cyan] {next(model.parameters()).device}")
print(f"[cyan]Model dtype:[/cyan] {next(model.parameters()).dtype}")

#%%
# ============================================================================
# CELL 2: Explore Model Configuration
# ============================================================================
# Investigate the model's configuration and architecture

print("\n[bold cyan]=" * 80)
print("[bold]MODEL CONFIGURATION EXPLORATION[/bold]")
print("=" * 80)

config = model.config
print(f"\n[bold]Model Type:[/bold] {config.model_type}")
print(f"[bold]Architecture:[/bold] {config.architectures}")

# Key configuration parameters
print("\n[bold yellow]Key Parameters:[/bold yellow]")
key_params = [
    'vocab_size', 'hidden_size', 'num_hidden_layers', 
    'num_attention_heads', 'intermediate_size', 'max_position_embeddings'
]
for param in key_params:
    if hasattr(config, param):
        value = getattr(config, param)
        print(f"  {param}: {value:,}")

# Explore: Print the full config to see all available parameters
print("\n[bold yellow]Full Configuration (explore in variable inspector):[/bold yellow]")
print("  config = model.config")
print("  # Check config in VS Code variable panel or print(config)")

#%%
# ============================================================================
# CELL 3: Explore Model Architecture Structure
# ============================================================================
# Understand the internal structure of the model

print("\n[bold cyan]=" * 80)
print("[bold]MODEL ARCHITECTURE STRUCTURE[/bold]")
print("=" * 80)

# List top-level components
print("\n[bold yellow]Top-level Components:[/bold yellow]")
for name, module in model.named_children():
    print(f"  {name}: {type(module).__name__}")

# Check for Thinker component
if hasattr(model, 'thinker'):
    print("\n[bold green]✓ Thinker component found![/bold green]")
    thinker = model.thinker
    print(f"  Type: {type(thinker).__name__}")
    
    # Explore Thinker's structure
    print("\n[bold yellow]Thinker Components:[/bold yellow]")
    for name, module in thinker.named_children():
        print(f"    {name}: {type(module).__name__}")
    
    # Thinker config
    if hasattr(thinker, 'config'):
        thinker_config = thinker.config
        print("\n[bold yellow]Thinker Configuration:[/bold yellow]")
        print(f"    Hidden size: {thinker_config.hidden_size}")
        print(f"    Num layers: {thinker_config.num_hidden_layers}")
        print(f"    Num attention heads: {thinker_config.num_attention_heads}")

# Explore: Check if Talker exists (it shouldn't in Thinker-only model)
if hasattr(model, 'talker'):
    print("\n[bold green]✓ Talker component found![/bold green]")
else:
    print("\n[yellow]ℹ Talker component not found (expected for Thinker-only model)[/yellow]")

#%%
# ============================================================================
# CELL 4: Explore Processor and Tokenizer
# ============================================================================
# Understand how inputs are processed

print("\n[bold cyan]=" * 80)
print("[bold]PROCESSOR AND TOKENIZER EXPLORATION[/bold]")
print("=" * 80)

print(f"\n[bold]Processor Type:[/bold] {type(processor).__name__}")

# Check processor components
if hasattr(processor, 'tokenizer'):
    tokenizer = processor.tokenizer
    print(f"\n[bold green]✓ Tokenizer found[/bold green]")
    print(f"  Type: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  Model max length: {tokenizer.model_max_length}")
    
    # Special tokens
    print("\n[bold yellow]Special Tokens:[/bold yellow]")
    special_tokens = {
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token,
        'bos_token': tokenizer.bos_token,
        'unk_token': tokenizer.unk_token,
    }
    for name, token in special_tokens.items():
        if token:
            print(f"  {name}: '{token}' (ID: {getattr(tokenizer, f'{name}_id')})")

# Check for image processor
if hasattr(processor, 'image_processor'):
    img_proc = processor.image_processor
    if img_proc is not None:
        print("\n[bold green]✓ Image processor found[/bold green]")
        print(f"  Type: {type(img_proc).__name__}")
        # Explore image processor config
        if hasattr(img_proc, 'size'):
            print(f"  Image size: {img_proc.size}")

# Check for audio processor
if hasattr(processor, 'audio_processor'):
    audio_proc = processor.audio_processor
    if audio_proc is not None:
        print("\n[bold green]✓ Audio processor found[/bold green]")
        print(f"  Type: {type(audio_proc).__name__}")

#%%
# ============================================================================
# CELL 5: Test Text-Only Input
# ============================================================================
# Simple text generation to understand the basic flow

print("\n[bold cyan]=" * 80)
print("[bold]TEXT-ONLY GENERATION TEST[/bold]")
print("=" * 80)

# Create a simple conversation
conversations = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a helpful AI assistant."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is machine learning in one sentence?"}
        ],
    },
]

print("\n[bold yellow]Input Conversation:[/bold yellow]")
print(f"  System: {conversations[0]['content'][0]['text']}")
print(f"  User: {conversations[1]['content'][0]['text']}")

# Process the conversation
inputs = processor.apply_chat_template(
    conversations,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
print(f"\n[cyan]Input shape:[/cyan] {inputs['input_ids'].shape}")

# Move to model device
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate
print("\n[bold yellow]Generating response...[/bold yellow]")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

# Decode
text = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(f"\n[bold green]Generated Text:[/bold green]")
print(text[0])

# Explore: Check the token IDs
print("\n[bold yellow]Token IDs (first 20):[/bold yellow]")
print(f"  Input IDs: {inputs['input_ids'][0][:20].tolist()}")
print(f"  Output IDs: {outputs[0][:20].tolist()}")

#%%
# ============================================================================
# CELL 6: Explore Model Parameters and Memory
# ============================================================================
# Understand model size and parameter count

print("\n[bold cyan]=" * 80)
print("[bold]MODEL PARAMETERS AND MEMORY[/bold]")
print("=" * 80)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n[bold yellow]Parameter Count:[/bold yellow]")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Estimate model size
dtype_size = 2 if next(model.parameters()).dtype in [torch.float16, torch.bfloat16] else 4
model_size_gb = total_params * dtype_size / 1e9
print(f"\n[bold yellow]Model Size:[/bold yellow]")
print(f"  Approximate size: {model_size_gb:.2f} GB ({dtype_size * 8}-bit)")

# Explore: Check parameter distribution by component
print("\n[bold yellow]Parameters by Component (explore further):[/bold yellow]")
if hasattr(model, 'thinker'):
    thinker_params = sum(p.numel() for p in model.thinker.parameters())
    print(f"  Thinker: {thinker_params:,} ({thinker_params/total_params*100:.1f}%)")

# Explore: Print specific layer parameters
print("\n[bold yellow]Explore specific layers:[/bold yellow]")
print("  # Example: Check embedding layer")
if hasattr(model, 'thinker') and hasattr(model.thinker, 'embed_tokens'):
    embed_params = model.thinker.embed_tokens.weight.numel()
    print(f"  Embedding layer: {embed_params:,} parameters")

#%%
# ============================================================================
# CELL 7: Explore Attention Mechanisms
# ============================================================================
# Investigate attention layers and their configurations

print("\n[bold cyan]=" * 80)
print("[bold]ATTENTION MECHANISMS EXPLORATION[/bold]")
print("=" * 80)

if hasattr(model, 'thinker') and hasattr(model.thinker, 'layers'):
    layers = model.thinker.layers
    print(f"\n[bold yellow]Number of layers:[/bold yellow] {len(layers)}")
    
    # Examine first layer
    if len(layers) > 0:
        first_layer = layers[0]
        print(f"\n[bold yellow]First Layer Structure:[/bold yellow]")
        for name, module in first_layer.named_children():
            print(f"  {name}: {type(module).__name__}")
        
        # Check attention config
        if hasattr(first_layer, 'self_attn'):
            attn = first_layer.self_attn
            print(f"\n[bold yellow]Self-Attention Configuration:[/bold yellow]")
            print(f"  Type: {type(attn).__name__}")
            if hasattr(attn, 'num_heads'):
                print(f"  Num heads: {attn.num_heads}")
            if hasattr(attn, 'num_key_value_heads'):
                print(f"  Key-value heads: {attn.num_key_value_heads}")
            if hasattr(attn, 'head_dim'):
                print(f"  Head dimension: {attn.head_dim}")

# Explore: Check attention implementation
print(f"\n[bold yellow]Attention Implementation:[/bold yellow]")
if hasattr(model.config, 'attn_implementation'):
    print(f"  {model.config.attn_implementation}")

#%%
# ============================================================================
# CELL 8: Test with Different Generation Parameters
# ============================================================================
# Experiment with generation settings

print("\n[bold cyan]=" * 80)
print("[bold]EXPERIMENTING WITH GENERATION PARAMETERS[/bold]")
print("=" * 80)

# Same conversation as before
conversations = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Tell me a short story about a robot."}
        ],
    },
]

inputs = processor.apply_chat_template(
    conversations,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Test 1: Greedy decoding (default)
print("\n[bold yellow]Test 1: Greedy Decoding (do_sample=False)[/bold yellow]")
with torch.no_grad():
    outputs_greedy = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )
text_greedy = processor.batch_decode(outputs_greedy, skip_special_tokens=True)[0]
print(f"  Output: {text_greedy}")

# Test 2: Sampling
print("\n[bold yellow]Test 2: Sampling (do_sample=True)[/bold yellow]")
with torch.no_grad():
    outputs_sample = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
text_sample = processor.batch_decode(outputs_sample, skip_special_tokens=True)[0]
print(f"  Output: {text_sample}")

# Explore: Try different parameters
print("\n[bold yellow]Try modifying these parameters:[/bold yellow]")
print("  - temperature: Controls randomness (0.1-2.0)")
print("  - top_p: Nucleus sampling threshold (0.0-1.0)")
print("  - top_k: Top-k sampling (number of tokens)")
print("  - max_new_tokens: Maximum tokens to generate")

#%%
# ============================================================================
# CELL 9: Explore Hidden States and Intermediate Outputs
# ============================================================================
# Investigate model internals during forward pass

print("\n[bold cyan]=" * 80)
print("[bold]EXPLORING HIDDEN STATES[/bold]")
print("=" * 80)

# Simple input
conversations = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello!"}
        ],
    },
]

inputs = processor.apply_chat_template(
    conversations,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Forward pass with output_hidden_states
print("\n[bold yellow]Forward pass with hidden states...[/bold yellow]")
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

# Explore hidden states
if hasattr(outputs, 'hidden_states'):
    hidden_states = outputs.hidden_states
    print(f"\n[bold green]✓ Hidden states available[/bold green]")
    print(f"  Number of layers: {len(hidden_states)}")
    print(f"  Shape of first hidden state: {hidden_states[0].shape}")
    print(f"  Shape of last hidden state: {hidden_states[-1].shape}")
    
    # Compare first and last
    print("\n[bold yellow]Hidden State Statistics:[/bold yellow]")
    print(f"  First layer - Mean: {hidden_states[0].mean().item():.6f}, Std: {hidden_states[0].std().item():.6f}")
    print(f"  Last layer  - Mean: {hidden_states[-1].mean().item():.6f}, Std: {hidden_states[-1].std().item():.6f}")

# Explore: Check logits
if hasattr(outputs, 'logits'):
    logits = outputs.logits
    print(f"\n[bold yellow]Logits Shape:[/bold yellow] {logits.shape}")
    print(f"  Logits statistics - Mean: {logits.mean().item():.6f}, Std: {logits.std().item():.6f}")

#%%
# ============================================================================
# CELL 10: Load Full Qwen2.5-Omni Model (with Talker)
# ============================================================================
# Compare Thinker-only vs Full model

print("\n[bold cyan]=" * 80)
print("[bold]LOADING FULL MODEL WITH TALKER[/bold]")
print("=" * 80)

print("\n[yellow]Note: This model includes the Talker component for audio generation.[/yellow]")
print("[yellow]It requires more memory than the Thinker-only version.[/yellow]")

# Load full model
print(f"\n[bold yellow]Loading: {model_name} (Full model)[/bold yellow]")
model_full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
print("[green]✓ Full model loaded[/green]")

# Compare components
print("\n[bold yellow]Model Components Comparison:[/bold yellow]")
print("\n[bold]Thinker-only model:[/bold]")
thinker_components = [name for name, _ in model.named_children()]
print(f"  Components: {thinker_components}")

print("\n[bold]Full model:[/bold]")
full_components = [name for name, _ in model_full.named_children()]
print(f"  Components: {full_components}")

# Check Talker component
if hasattr(model_full, 'talker'):
    print("\n[bold green]✓ Talker component found in full model![/bold green]")
    talker = model_full.talker
    print(f"  Type: {type(talker).__name__}")
    
    if hasattr(talker, 'config'):
        talker_config = talker.config
        print("\n[bold yellow]Talker Configuration:[/bold yellow]")
        print(f"    Hidden size: {talker_config.hidden_size}")
        print(f"    Num layers: {talker_config.num_hidden_layers}")

# Compare parameter counts
thinker_params = sum(p.numel() for p in model.parameters())
full_params = sum(p.numel() for p in model_full.parameters())
print(f"\n[bold yellow]Parameter Comparison:[/bold yellow]")
print(f"  Thinker-only: {thinker_params:,}")
print(f"  Full model: {full_params:,}")
print(f"  Difference (Talker): {full_params - thinker_params:,}")

#%%
# ============================================================================
# CELL 11: Test Full Model with Thinker/Talker Parameters
# ============================================================================
# Understand how Thinker and Talker work together

print("\n[bold cyan]=" * 80)
print("[bold]THINKER-TALKER ARCHITECTURE TEST[/bold]")
print("=" * 80)

conversations = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is artificial intelligence?"}
        ],
    },
]

inputs = processor.apply_chat_template(
    conversations,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(model_full.device) for k, v in inputs.items()}

print("\n[bold yellow]Generating with separate Thinker/Talker parameters...[/bold yellow]")
print("[yellow]Note: thinker_* parameters control text generation[/yellow]")
print("[yellow]      talker_* parameters control audio generation[/yellow]")

with torch.no_grad():
    outputs = model_full.generate(
        **inputs,
        max_new_tokens=100,
        thinker_do_sample=False,  # Thinker: greedy decoding
        talker_do_sample=False,   # Talker: greedy decoding
    )

# Full model may return tuple (text_ids, audio) or just text_ids
if isinstance(outputs, tuple):
    text_ids, audio = outputs
    print(f"\n[bold green]✓ Got both text and audio outputs![/bold green]")
    print(f"  Text IDs shape: {text_ids.shape}")
    print(f"  Audio shape: {audio.shape}")
    
    text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
    print(f"\n[bold]Generated Text:[/bold] {text}")
    
    if audio is not None and HAS_SOUNDFILE:
        print(f"\n[bold yellow]Audio can be saved:[/bold yellow]")
        print("  sf.write('output.wav', audio.reshape(-1).cpu().numpy(), samplerate=24000)")
else:
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\n[bold]Generated Text:[/bold] {text}")

#%%
# ============================================================================
# CELL 12: Load Qwen2-Audio Model
# ============================================================================
# Explore the audio-focused model

print("\n[bold cyan]=" * 80)
print("[bold]LOADING QWEN2-AUDIO MODEL[/bold]")
print("=" * 80)

audio_model_name = "Qwen/Qwen2-Audio-7B-Instruct"
print(f"\n[bold yellow]Loading: {audio_model_name}[/bold yellow]")

# Load processor
audio_processor = AutoProcessor.from_pretrained(
    audio_model_name,
    cache_dir=MODEL_CACHE_DIR
)
print("[green]✓ Processor loaded[/green]")

# Load model
audio_model = AutoModelForSeq2SeqLM.from_pretrained(
    audio_model_name,
    cache_dir=MODEL_CACHE_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
print("[green]✓ Model loaded[/green]")

# Compare architectures
print("\n[bold yellow]Architecture Comparison:[/bold yellow]")
print(f"  Qwen2.5-Omni: {type(model).__name__}")
print(f"  Qwen2-Audio: {type(audio_model).__name__}")

# Check config
audio_config = audio_model.config
print("\n[bold yellow]Qwen2-Audio Configuration:[/bold yellow]")
print(f"  Model type: {audio_config.model_type}")
print(f"  Hidden size: {audio_config.hidden_size}")
print(f"  Num layers: {audio_config.num_hidden_layers}")

#%%
# ============================================================================
# CELL 13: Test Qwen2-Audio with Text Input
# ============================================================================
# Test the audio model's text understanding

print("\n[bold cyan]=" * 80)
print("[bold]TESTING QWEN2-AUDIO WITH TEXT[/bold]")
print("=" * 80)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is audio processing?"}
        ],
    },
]

inputs = audio_processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(audio_model.device) for k, v in inputs.items()}

print(f"\n[cyan]Input shape:[/cyan] {inputs['input_ids'].shape}")

with torch.no_grad():
    outputs = audio_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

# Decode (trim input tokens)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], outputs)
]
text = audio_processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)
print(f"\n[bold green]Generated Text:[/bold green]")
print(text[0])

#%%
# ============================================================================
# CELL 14: Explore Model Differences
# ============================================================================
# Compare the different Qwen models

print("\n[bold cyan]=" * 80)
print("[bold]MODEL COMPARISON[/bold]")
print("=" * 80)

comparison_table = Table(title="Qwen Models Comparison")
comparison_table.add_column("Model", style="cyan")
comparison_table.add_column("Parameters", style="green")
comparison_table.add_column("Modalities", style="yellow")
comparison_table.add_column("Outputs", style="magenta")
comparison_table.add_column("Architecture", style="blue")

# Get parameter counts
thinker_params = sum(p.numel() for p in model.parameters())
full_params = sum(p.numel() for p in model_full.parameters())
audio_params = sum(p.numel() for p in audio_model.parameters())

comparison_table.add_row(
    "Qwen2.5-Omni Thinker",
    f"{thinker_params/1e9:.1f}B",
    "Text, Image, Audio, Video",
    "Text",
    "Thinker-Talker (Thinker only)"
)
comparison_table.add_row(
    "Qwen2.5-Omni Full",
    f"{full_params/1e9:.1f}B",
    "Text, Image, Audio, Video",
    "Text, Audio",
    "Thinker-Talker (Full)"
)
comparison_table.add_row(
    "Qwen2-Audio",
    f"{audio_params/1e9:.1f}B",
    "Text, Audio",
    "Text",
    "Seq2Seq"
)

console.print(comparison_table)

#%%
# ============================================================================
# CELL 15: Helper Functions for Testing with Files
# ============================================================================
# Functions to test with actual image/audio/video files

def test_with_image(image_path, model, processor, question="Describe this image."):
    """Test model with an image file"""
    if not HAS_PIL:
        print("[red]PIL/Pillow not available[/red]")
        return None
    
    if not os.path.exists(image_path):
        print(f"[red]Image file not found: {image_path}[/red]")
        return None
    
    try:
        image = Image.open(image_path)
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ],
            },
        ]
        
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        
        text = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text[0]
        
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        return None

def test_with_audio(audio_path, model, processor, question="What did you hear?"):
    """Test model with an audio file"""
    if not os.path.exists(audio_path):
        print(f"[red]Audio file not found: {audio_path}[/red]")
        return None
    
    try:
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": question}
                ],
            },
        ]
        
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        
        text = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text[0]
        
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        return None

print("\n[bold green]✓ Helper functions defined[/bold green]")
print("\n[yellow]Usage examples:[/yellow]")
print("  # Test with image:")
print("  # result = test_with_image('path/to/image.jpg', model, processor)")
print("  # Test with audio:")
print("  # result = test_with_audio('path/to/audio.wav', model, processor)")

#%%
# ============================================================================
# CELL 16: Explore Further - Custom Experiments
# ============================================================================
# Space for your own experiments!

print("\n[bold cyan]=" * 80)
print("[bold]YOUR EXPERIMENTS[/bold]")
print("=" * 80)

print("\n[yellow]This cell is for your own exploration![/yellow]")
print("\n[bold]Ideas to try:[/bold]")
print("  1. Modify generation parameters (temperature, top_p, top_k)")
print("  2. Test with different conversation formats")
print("  3. Explore attention weights (output_attentions=True)")
print("  4. Compare outputs between Thinker-only and Full model")
print("  5. Test with actual image/audio files using helper functions")
print("  6. Investigate token embeddings")
print("  7. Explore layer-wise outputs")

print("\n[bold]Example:[/bold]")
print("  # Test different temperature values")
print("  # for temp in [0.1, 0.7, 1.0, 1.5]:")
print("  #     outputs = model.generate(..., temperature=temp)")
print("  #     print(f'Temperature {temp}: ...')")

#%%
