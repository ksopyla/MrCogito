#%%
"""
Qwen Omni & Audio Exploration Guide
===================================
Interactive playground to explore Qwen2.5 Omni, Qwen3 Omni, and Qwen2 Audio.
Focuses on usage, modalities (Audio/Text), inputs/outputs, and architecture components.

Key Models:
- Qwen2.5-Omni (7B/3B): Omniverse model (Text/Audio input -> Text/Audio output)
- Qwen2-Audio: Audio/Text input -> Text output
- Qwen3-Omni (Preview): Next-gen multimodal
"""

import os
import torch
import soundfile as sf
import numpy as np
from PIL import Image
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoProcessor, 
    TextIteratorStreamer
)
import librosa

# Cache directory configuration
MODEL_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Cache", "Models"))
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
print(f"Model cache: {MODEL_CACHE_DIR}")

#%%
# ============================================================================
# CELL 1: Model Selection & Configuration
# ============================================================================
# Define available models and their capabilities

MODELS = {
    "qwen2.5_omni": {
        "id": "Qwen/Qwen2.5-Omni-7B",
        "type": "omni",
        "description": "Multimodal (Audio/Text In -> Audio/Text Out)"
    },
    "qwen2_audio": {
        "id": "Qwen/Qwen2-Audio-7B-Instruct",
        "type": "audio_text",
        "description": "Audio Understanding (Audio/Text In -> Text Out)"
    },
    # Note: Qwen3 is bleeding edge, might require specific access or libraries
    "qwen3_omni": {
        "id": "Qwen/Qwen3-Omni-30B-A3B-Instruct", 
        "type": "omni",
        "description": "Advanced Multimodal (Mixture of Experts)"
    }
}

# Select model to load for this session
# Change this to 'qwen2_audio' or 'qwen3_omni' to switch
SELECTED_MODEL_KEY = "qwen2.5_omni" 
model_config = MODELS[SELECTED_MODEL_KEY]

print(f"\n🎯 === Selected Model ===")
print(f"Model: {model_config['id']}")
print(f"Type: {model_config['type']}")

#%%
# ============================================================================
# CELL 1.5: Create Dummy Data for Examples
# ============================================================================
# Create all dummy/test data upfront with explanations.
# This data will be reused throughout the script for various examples.

print("\n🎨 === Creating Dummy Data for Examples ===")
print("All dummy data is created here and reused in later cells.\n")

# ============================================================================
# IMAGE DATA
# ============================================================================
print("📸 Image Data:")
print("   - Format: PIL Image objects")
print("   - Size: 224x224 pixels (standard vision model input size)")
print("   - Color space: RGB (3 channels)")
print("   - Purpose: Simulate image inputs for multimodal examples\n")

# Red image for single image example
dummy_image_red = Image.new('RGB', (224, 224), color='red')
print(f"   ✅ dummy_image_red: {dummy_image_red.size} RGB image (red)")

# Blue image for multimodal example
dummy_image_blue = Image.new('RGB', (224, 224), color='blue')
print(f"   ✅ dummy_image_blue: {dummy_image_blue.size} RGB image (blue)")

# ============================================================================
# AUDIO DATA
# ============================================================================
print("\n🎵 Audio Data:")
print("   - Format: NumPy arrays (float32)")
print("   - Sampling rate: 16,000 Hz (standard for speech models)")
print("   - Duration: 1 second each")
print("   - Shape: (16000,) - mono channel audio")
print("   - Purpose: Simulate audio inputs for multimodal examples\n")

# Audio 1: 440 Hz sine wave (musical note A4)
# Formula: sin(2π × frequency × time)
sample_rate = 16000
duration = 1.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
dummy_audio_440hz = np.sin(2 * np.pi * 440 * t).astype(np.float32)
print(f"   ✅ dummy_audio_440hz: {dummy_audio_440hz.shape} array, 440 Hz sine wave (A4 note)")
print(f"      Range: [{dummy_audio_440hz.min():.3f}, {dummy_audio_440hz.max():.3f}]")

# Audio 2: 880 Hz sine wave (musical note A5, one octave higher)
dummy_audio_880hz = np.sin(2 * np.pi * 880 * t).astype(np.float32)
print(f"   ✅ dummy_audio_880hz: {dummy_audio_880hz.shape} array, 880 Hz sine wave (A5 note)")
print(f"      Range: [{dummy_audio_880hz.min():.3f}, {dummy_audio_880hz.max():.3f}]")

# Audio 3: Random noise (for different example)
dummy_audio_noise = np.random.uniform(-0.5, 0.5, int(sample_rate * duration)).astype(np.float32)
print(f"   ✅ dummy_audio_noise: {dummy_audio_noise.shape} array, random uniform noise")
print(f"      Range: [{dummy_audio_noise.min():.3f}, {dummy_audio_noise.max():.3f}]")

print("\n💡 Note: Real audio files would be loaded using librosa.load() or soundfile.read()")
print("   Example: audio, sr = librosa.load('path/to/audio.wav', sr=16000)")

#%%
# ============================================================================
# CELL 2.1: Loading Tokenizer & Processor
# ============================================================================
# Explore the tokenizer and processor structure before loading the heavy model.

print(f"Loading processor for {model_config['id']}...")

processor = AutoProcessor.from_pretrained(
    model_config['id'], 
    trust_remote_code=True,
    cache_dir=MODEL_CACHE_DIR
)
tokenizer = processor.tokenizer
print("✅ Processor and Tokenizer loaded successfully")

#%%
# ============================================================================
# CELL 2.2: Inspect Special Tokens
# ============================================================================

print("\n🔍 === Special Tokens Map ===")
print(tokenizer.special_tokens_map)

print("\n🔍 === Additional Special Tokens (Audio/Image) ===")
# Check for common multimodal tokens
known_special_tokens = ["<|audio_bos|>", "<|audio_eos|>", "<|image_pad|>", "<|video_pad|>"]
for token in known_special_tokens:
    if token in tokenizer.get_vocab():
        print(f"  {token}: {tokenizer.convert_tokens_to_ids(token)}")
    else:
        print(f"  {token}: Not found")

#%%
# ============================================================================
# CELL 2.3: Inspect Chat Template
# ============================================================================

print("\n🔍 === Chat Template ===")
if tokenizer.chat_template:
    print(f"Template preview: \n{tokenizer.chat_template}")
else:
    print("⚠️ WARNING: No chat template found in tokenizer config")



#%%
# ============================================================================
# CELL 2.4: Example 1 - Text-Only Chat
# ============================================================================

print("\n📝 === Example 1: TEXT-ONLY CHAT ===")
print("Multi-turn conversation with system, user, and assistant messages\n")

# Note: Qwen2.5 Omni processor expects a system message as the first message
# If omitted, it will warn that audio output may not work as expected
messages_text_only = [
    {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
    {"role": "user", "content": [{"type": "text", "text": "What is machine learning?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Machine learning is a subset of AI..."}]},
    {"role": "user", "content": [{"type": "text", "text": "Can you give me an example?"}]}
]

formatted_text = processor.apply_chat_template(
    messages_text_only,
    add_generation_prompt=True,
    tokenize=False
)
print("Formatted prompt:")
print("-" * 80)
print(formatted_text)
print("-" * 80)

#%%
# ============================================================================
# CELL 2.5: Example 2 - Chat with Image
# ============================================================================

print("\n📝 === Example 2: CHAT WITH IMAGE ===")
print("Image + text question\n")

messages_with_image = [
    {"role": "user", "content": [
        {"type": "image", "image": dummy_image_red},
        {"type": "text", "text": "What's in this image?"}
    ]}
]

formatted_image = processor.apply_chat_template(
    messages_with_image,
    add_generation_prompt=True,
    tokenize=False
)
print("Formatted prompt:")
print("-" * 80)
print(formatted_image)
print("-" * 80)
print("Note: Image is represented as special tokens or embeddings in the actual prompt")

#%%
# ============================================================================
# CELL 2.6: Example 3 - Chat with Audio
# ============================================================================

print("\n📝 === Example 3: CHAT WITH AUDIO ===")
print("Audio + text question\n")

messages_with_audio = [
    {"role": "user", "content": [
        {"type": "audio", "audio": dummy_audio_440hz},
        {"type": "text", "text": "What did you hear in this audio?"}
    ]}
]

formatted_audio = processor.apply_chat_template(
    messages_with_audio,
    add_generation_prompt=True,
    tokenize=False
)
print("Formatted prompt:")
print("-" * 80)
print(formatted_audio)
print("-" * 80)
print("Note: Audio is processed into embeddings/tokens before being included")

#%%
# ============================================================================
# CELL 2.7: Example 4 - Multimodal Chat (Text + Image + Audio)
# ============================================================================

print("\n📝 === Example 4: MULTIMODAL CHAT (Text + Image + Audio) ===")
print("Combining multiple modalities in one message\n")

messages_multimodal = [
    {"role": "user", "content": [
        {"type": "text", "text": "I have an image and audio:"},
        {"type": "image", "image": dummy_image_blue},
        {"type": "audio", "audio": dummy_audio_880hz},
        {"type": "text", "text": "Can you describe both?"}
    ]}
]

formatted_multimodal = processor.apply_chat_template(
    messages_multimodal,
    add_generation_prompt=True,
    tokenize=False
)
print("Formatted prompt:")
print("-" * 80)
print(formatted_multimodal)
print("-" * 80)

#%%
# ============================================================================
# CELL 2.8: Example 5 - Chat with Video
# ============================================================================

print("\n📝 === Example 5: CHAT WITH VIDEO ===")
print("Video + text question (if supported)\n")

messages_with_video = [
    {"role": "user", "content": [
        {"type": "video", "video": "path/to/video.mp4"},
        {"type": "text", "text": "What happens in this video?"}
    ]}
]

formatted_video = processor.apply_chat_template(
    messages_with_video,
    add_generation_prompt=True,
    tokenize=False
)
print("Formatted prompt:")
print("-" * 80)
print(formatted_video)
print("-" * 80)

#%%
# ============================================================================
# CELL 2.9: Tokenized Input Examples
# ============================================================================

def analyze_tokenized_messages(messages, title="Tokenized Input"):
    """
    Analyze and print detailed tokenization information for a set of messages.
    
    Args:
        messages: List of message dictionaries (same format as for apply_chat_template)
        title: Title to display for this analysis
    
    Note: Uses the global 'processor' and 'tokenizer' variables (loaded in earlier cells)
    """
    print(f"\n{title}:")
    print("=" * 80)
    
    # Tokenize the messages (uses global processor and tokenizer)
    tokenized = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )
    
    # tokenized is a torch.Tensor with shape [1, seq_len]
    input_ids = tokenized[0]  # Get first sequence
    
    print(f"  Shape: {tokenized.shape}")
    print(f"  Sequence length: {len(input_ids)} tokens")
    print(f"\n  All token IDs: {input_ids.tolist()}")
    print(f"\n  Full decoded text:")
    print(f"  {'-' * 76}")
    decoded_full = tokenizer.decode(input_ids)
    print(f"  {decoded_full}")
    print(f"  {'-' * 76}")
    
    print(f"\n  Token-by-token breakdown:")
    for i, token_id in enumerate(input_ids):
        token_str = tokenizer.decode([token_id])
        # Clean up token string for display
        token_str = token_str.replace('\n', '\\n').replace('\r', '\\r')
        print(f"    [{i:3d}] ID: {token_id.item():6d} → '{token_str}'")
    
    # Show special tokens in the sequence
    special_token_ids = set()
    special_token_names = {}
    for token_name in ['bos_token_id', 'eos_token_id', 'pad_token_id']:
        if hasattr(tokenizer, token_name):
            token_id = getattr(tokenizer, token_name)
            if token_id is not None:
                special_token_ids.add(token_id)
                special_token_names[token_id] = token_name.replace('_token_id', '')
    
    # Also check for multimodal special tokens
    for token_str in ['<|im_start|>', '<|im_end|>', '<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>', 
                      '<|vision_bos|>', '<|vision_eos|>', '<|audio_bos|>', '<|audio_eos|>']:
        if token_str in tokenizer.get_vocab():
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            special_token_ids.add(token_id)
            special_token_names[token_id] = token_str
    
    found_special = [(idx, tid.item(), special_token_names.get(tid.item(), 'unknown')) 
                     for idx, tid in enumerate(input_ids) if tid.item() in special_token_ids]
    if found_special:
        print(f"\n  Special token positions:")
        for idx, tid, name in found_special:
            print(f"    Position [{idx:3d}]: ID {tid:6d} ({name})")
    
    print("=" * 80)
    return tokenized, input_ids

# ============================================================================
# 🔧 CONFIGURATION: Change this variable to test different message types
# ============================================================================
# Available options:
#   - messages_text_only      : Text-only conversation
#   - messages_with_image     : Image + text
#   - messages_with_audio     : Audio + text
#   - messages_multimodal      : Text + Image + Audio
#   - messages_with_video     : Video + text
#   - messages_text            : Simple text (from CELL 5)
#   - messages_audio           : Audio example (from CELL 5)

MESSAGES_TO_ANALYZE = messages_with_audio  # 👈 Change this line to test different message types!

# ============================================================================
# Run the analysis
# ============================================================================
print("\n🔢 === Tokenized Input Examples ===")
print("Change 'MESSAGES_TO_ANALYZE' variable above to test different message types!\n")

tokenized_output, token_ids = analyze_tokenized_messages(MESSAGES_TO_ANALYZE)

# Example: Show what happens with multimodal inputs
print("\n💡 Multimodal Input Structure Info:")
print("   When processing multimodal inputs, the processor typically:")
print("   - Converts images to embeddings/patches")
print("   - Converts audio to spectrograms or embeddings")
print("   - Interleaves these with text tokens using special tokens")
print("   - Creates attention masks that account for all modalities")
print("\n   Example token sequence structure:")
print("   [BOS] [text_tokens...] [IMAGE_START] [image_tokens...] [IMAGE_END] [text_tokens...] [AUDIO_START] [audio_tokens...] [AUDIO_END] [text_tokens...] [EOS]")

#%%
# ============================================================================
# CELL 3: Load the Model
# ============================================================================
# Loading the model with trust_remote_code=True to handle custom architectures (Thinker/Talker)

print(f"Loading model {model_config['id']}...")
print("⚡ (This may take a while and consume significant VRAM...)")

model = AutoModelForCausalLM.from_pretrained(
    model_config['id'],
    trust_remote_code=True,
    torch_dtype=torch.float16, # Use bfloat16 if supported by GPU
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR
)
print(f"✅ Model loaded: {type(model).__name__}")

#%%
# ============================================================================
# CELL 4: Architecture Exploration (Thinker vs Talker)
# ============================================================================
# Omni models often separate "Reasoning" (Thinker) from "Speech Generation" (Talker).
# Let's inspect the model structure to find these components.

print("\n🔍 === Model Architecture / Top-Level Modules ===")
for name, module in model.named_children():
    print(f"  - {name}: {type(module).__name__}")

# Check for specific Omni components
has_thinker = hasattr(model, 'thinker') or any('thinker' in n for n, _ in model.named_children())
has_talker = hasattr(model, 'talker') or any('talker' in n for n, _ in model.named_children())

if has_thinker:
    print("\n✅ Thinker Component Found - Handles text reasoning and generation.")
    # If exposed as a property or attribute
    if hasattr(model, 'thinker'):
        print(f"  Config: {model.thinker.config if hasattr(model.thinker, 'config') else 'N/A'}")

if has_talker:
    print("\n✅ Talker Component Found - Handles audio/speech synthesis.")
    if hasattr(model, 'talker'):
        print(f"  Config: {model.talker.config if hasattr(model.talker, 'config') else 'N/A'}")

#%%
# ============================================================================
# CELL 5: Input Processing & Chat Templates
# ============================================================================
# How to prepare inputs for the model (Text + Audio)

# 1. Text Only Example
messages_text = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "text", "text": "Hello, how does a transformer work?"}]}
]

text_input = processor.apply_chat_template(
    messages_text, 
    add_generation_prompt=True, 
    tokenize=False # Just to see the string
)
print("\n💬 === Formatted Text Prompt ===")
print(text_input)

# 2. Audio Input Example
# Note: Using dummy_audio_noise from CELL 1.5 (created at the beginning)
# For template formatting only, we use a placeholder string
# For actual processing, use the numpy array from dummy_audio_noise
messages_audio = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [
        {"type": "audio", "audio": dummy_audio_noise},  # Using centralized dummy data
        {"type": "text", "text": "What is this sound?"}
    ]}
]

# Note: apply_chat_template usually handles string formatting. 
# Actual tensor creation happens via processor(text=..., audios=...)

print("\n🔧 === Input Processing & Tensor Creation ===")
# This demonstrates how the processor converts formatted prompts to model inputs

# Show actual input tensor structure
print("\n1️⃣ Text-Only Input Processing:")
inputs_text = processor(
    text=[text_input],
    return_tensors="pt",
    padding=True
)
print("   Input dictionary keys:", list(inputs_text.keys()))
for key, value in inputs_text.items():
    if isinstance(value, torch.Tensor):
        print(f"   - {key}: shape {value.shape}, dtype {value.dtype}")
    else:
        print(f"   - {key}: {type(value)}")

# Show multimodal input processing (if supported)
print("\n2️⃣ Multimodal Input Processing Examples:")
print("   For models supporting multiple modalities, inputs may include:")

print("\n   📝 TEXT INPUT:")
print("   inputs = processor(text=[formatted_prompt], return_tensors='pt')")
print("   → Returns: {'input_ids': tensor, 'attention_mask': tensor}")

if "audio" in model_config['type'] or model_config['type'] == "omni":
    print("\n   🎵 AUDIO INPUT:")
    print("   inputs = processor(")
    print("       text=[formatted_prompt],")
    print("       audios=[audio_array],  # numpy array or list of arrays")
    print("       sampling_rate=16000,")
    print("       return_tensors='pt'")
    print("   )")
    print("   → Returns: {")
    print("       'input_ids': tensor,           # Text tokens")
    print("       'attention_mask': tensor,      # Attention mask")
    print("       'audio_values': tensor,        # Audio waveform or features")
    print("       'audio_attention_mask': tensor # Audio attention mask")
    print("   }")
    
    print("\n   📸 IMAGE INPUT:")
    print("   inputs = processor(")
    print("       text=[formatted_prompt],")
    print("       images=[pil_image],  # PIL Image or list of Images")
    print("       return_tensors='pt'")
    print("   )")
    print("   → Returns: {")
    print("       'input_ids': tensor,           # Text tokens")
    print("       'attention_mask': tensor,      # Attention mask")
    print("       'pixel_values': tensor,        # Image embeddings/patches")
    print("       'image_grid_thw': tensor       # Image grid dimensions")
    print("   }")
    
    print("\n   🎬 VIDEO INPUT:")
    print("   inputs = processor(")
    print("       text=[formatted_prompt],")
    print("       videos=[video_array],  # Video frames or path")
    print("       return_tensors='pt'")
    print("   )")
    print("   → Returns: Similar to image but with temporal dimension")

print("\n3️⃣ Input Shape Examples:")
print("   Typical shapes for Qwen Omni/Audio models:")
print("   - input_ids: [1, seq_len] where seq_len varies (e.g., 512-8192)")
print("   - attention_mask: [1, seq_len] (1 for real tokens, 0 for padding)")
print("   - audio_values: [1, audio_len] or [1, num_frames, feature_dim]")
print("   - pixel_values: [1, num_patches, patch_dim] for images")

print("\n💡 Note: Actual shapes depend on:")
print("   - Model architecture (Omni vs Audio vs standard)")
print("   - Input length (text + modalities)")
print("   - Model's max context length")
print("   - How modalities are encoded (patches, embeddings, etc.)")

#%%
# ============================================================================
# CELL 6: Generating Response (Thinker Mode)
# ============================================================================
# Generate text response (Thinker only)

print("\n⚡ === Generating Text Response ===")
inputs = processor(
    text=[text_input], 
    return_tensors="pt", 
    padding=True
)
inputs = inputs.to(model.device)

# Standard generation (Text-to-Text)
with torch.no_grad():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

decoded_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n✨ === Generated Output ===")
print(decoded_text)

#%%
# ============================================================================
# CELL 7: Omni Generation (Thinker + Talker) / Audio Output
# ============================================================================
# If the model supports audio output (Omni), we check specific generation parameters.

if model_config['type'] == "omni":
    print("\n🎵 === Attempting Audio Generation (Omni Mode) ===")
    
    # Many Omni models allow controlling audio generation via flags in .generate()
    # e.g., output_audio=True, or specific audio_params
    
    # We inspect the generate method to see if custom kwargs are supported or documented
    # (This is exploratory as API varies by specific Omni version)
    
    print("💡 Common Omni arguments: 'output_audio=True' or 'talker_do_sample=True'")
    
    # This assumes a hypothetical API common in recent Omni models
    # Adjust based on specific model documentation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            # Experimental flags often found in these models:
            # output_audio=True, 
            # return_dict_in_generate=True 
        )
        
    # Check if output contains audio (often in a separate field or extended tensor)
    if hasattr(outputs, 'audio') or isinstance(outputs, dict):
        print("✅ Audio output container found")
    else:
        print("⚠️ WARNING: Standard tensor output received. Audio generation might need specific flags or post-processing.")

#%%
# ============================================================================
# CELL 8: Audio Streaming Concept
# ============================================================================
# How to handle streaming audio I/O

print("\n🌊 === Audio Streaming Architecture ===")
print("""
Streaming with Omni models typically involves:
1. Input Stream: VAD (Voice Activity Detection) -> Accumulate chunks -> Feed to processor
2. Output Stream: 
   - Model generates tokens
   - Specific tokens trigger 'Talker'
   - Talker outputs latent audio codes or waveform chunks
   - Yield chunks to audio device
""")

# Example pseudo-code for streaming loop
print("\nPseudocode for streaming loop:")
print("""
streamer = TextIteratorStreamer(tokenizer)
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for new_text in streamer:
    print(new_text)
    # If audio is interleaved, extract and play
""")

#%%
# ============================================================================
# CELL 9: Deep Dive - Tokenizer & Vocabulary Statistics
# ============================================================================

vocab = tokenizer.get_vocab()
print(f"\n📊 === Vocabulary Size ===")
print(f"Vocabulary Size: {len(vocab)}")

# Check for audio specific token ranges if likely
# Qwen-Audio often uses specific ranges for audio patch tokens
audio_tokens = [k for k in vocab.keys() if "AUDIO" in k or "audio" in k]
print(f"Tokens containing 'AUDIO': {len(audio_tokens)}")
if len(audio_tokens) > 0:
    print(f"Samples: {audio_tokens[:10]}")

#%%

