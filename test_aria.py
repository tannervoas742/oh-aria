import torch
from PIL import Image
from transformers import AutoConfig, BitsAndBytesConfig
from aria.model import AriaForConditionalGeneration, AriaProcessor
from accelerate import Accelerator

# ==============================
# Configuration Switch
# ==============================
LOAD_TRAINED_MODEL = False  # Toggle: Set True for trained, False for untrained model
use_flash_attn = False  # Set True to enable flash attention 2
model_path = "rhymes-ai/Aria"

# ==============================
# Accelerator Initialization
# ==============================
accelerator = Accelerator()

# Helper Function: Extract model name
def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

# ==============================
# Load Pretrained Model Function
# ==============================
def load_pretrained_model(
    model_path,
    load_8bit=False,
    load_4bit=False,
    use_flash_attn=False,
    **kwargs
):
    # Quantization arguments
    kwargs = {}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    else:
        kwargs['attn_implementation'] = 'sdpa'

    # Load config, processor, and model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AriaProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AriaForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        **kwargs
    )
    # Prepare model for accelerator
    model = accelerator.prepare(model)
    return processor, model, config.get("max_sequence_length", 2048)

# ==============================
# Load Untrained Model Function
# ==============================
def load_untrained_model(model_path, use_flash_attn=False, **kwargs):
    # Load config and processor
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if use_flash_attn:
        config.attn_implementation = "flash_attention_2"
    else:
        config.attn_implementation = "sdpa"

    processor = AriaProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AriaForConditionalGeneration(config)

    # Move to Accelerator device
    model = accelerator.prepare(model)
    #context_len = config.get("max_sequence_length", 2048)
    return processor, model #, context_len

# ==============================
# Model Loading
# ==============================
if LOAD_TRAINED_MODEL:
    print("Loading PRETRAINED model...")
    processor, model, context_len = load_pretrained_model(
        model_path, load_4bit=False, use_flash_attn=use_flash_attn
    )
else:
    print("Loading UNTRAINED model...")
    processor, model = load_untrained_model(
        model_path, use_flash_attn=use_flash_attn
    )

# ==============================
# Input Preparation and Inference
# ==============================
image_path = "cat.png"
image = Image.open(image_path).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"text": None, "type": "image"},
            {"text": "What is the image?", "type": "text"},
        ],
    }
]

from habana_frameworks.torch.hpu import memory_summary
print(memory_summary())

# Prepare inputs
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt")
inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

# Inference
with torch.inference_mode(), accelerator.autocast():
    import pdb; pdb.set_trace()
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        stop_strings=["<|im_end|>"],
        tokenizer=processor.tokenizer,
        do_sample=True,
        temperature=0.9,
    )
    output_ids = output[0][inputs["input_ids"].shape[1]:]
    result = processor.decode(output_ids, skip_special_tokens=True)

print("Result:", result)
