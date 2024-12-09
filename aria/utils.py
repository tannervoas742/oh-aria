import torch
from PIL import Image
from transformers import AutoConfig
from aria.model import AriaForConditionalGeneration, AriaProcessor
from accelerate import Accelerator
from decord import VideoReader
from tqdm import tqdm
from typing import Any, List, Dict

# ==============================
# Accelerator Initialization
# ==============================
accelerator = Accelerator()

class ModelLoader:
    # ==============================
    # Load Pretrained Model
    # ==============================
    def load_pretrained_model(
        model_path: str,
        use_flash_attn: bool = False,
        accelerator: Accelerator = accelerator,
        **kwargs
    ):
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
        return processor, model

    # ==============================
    # Load Untrained Model Function
    # ==============================
    def load_untrained_model(
        model_path: str,
        use_flash_attn: bool =False,
        accelerator: Accelerator = accelerator,
        **kwargs
    ):
        kwargs['torch_dtype'] = torch.bfloat16

        if use_flash_attn:
            kwargs['attn_implementation'] = 'flash_attention_2'
        else:
            kwargs['attn_implementation'] = 'sdpa'

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
        return processor, model

# ==============================
# Processing Helper Functions
# ==============================
class InputHelper:
    def images_to_inputs(
        model: AriaForConditionalGeneration,
        processor: AriaProcessor,
        image_paths: List[str],
        query: str="Describe the images and analyze their similarities and differences in a JSON format.",
    ):
        images, contents = [], []
        for i, image_path in enumerate(image_paths):
            images.append(Image.open(image_path).convert("RGB"))
            contents.extend(
                [
                    {"text": f"Image {i+1}: ", "type": "text"},
                    {"text": None, "type": "image"},
                    {"text": "\n", "type": "text"},
                ]
            )

        messages = [
            {
                "role": "user",
                "content": [
                    *contents,
                    {"text": query, "type": "text"},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        return inputs
    
    def image_to_inputs(
        model: AriaForConditionalGeneration,
        processor: AriaProcessor,
        image_path: str,
        query: str="Describe the objects, their activity, and the scene of this image in details. Return a JSON dict.",
    ):
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": None, "type": "image"},
                    {"text": query, "type": "text"},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        return inputs
    
    def video_to_inputs(
        model: AriaForConditionalGeneration,
        processor: AriaProcessor,
        video_path: str,
        num_frames: int = 32,
        query: str="Describe what the videos shows in detail.",
    ):
        vr = VideoReader(video_path)
        duration = len(vr)
        frame_indices = [int(duration / num_frames * (i + 0.5)) for i in range(num_frames)]
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(frame).convert("RGB") for frame in frames]
        
        contents = []
        for i, frame in enumerate(frames):
            contents.extend(
                [
                    {"text": f"Frame {i+1}: ", "type": "text"},
                    {"text": None, "type": "image"},
                    {"text": "\n", "type": "text"},
                ]
            )

        messages = [
            {
                "role": "user",
                "content": [
                    *contents,
                    {"text": query, "type": "text"},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=frames, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        return inputs
    
def infer(
    inputs: Dict[Any, Any],
    model: AriaForConditionalGeneration,
    processor: AriaProcessor,
    use_flash_attn: bool = False,
    accelerator: Accelerator = accelerator,
):
    with torch.inference_mode(), accelerator.autocast():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            stop_strings=["<|im_end|>"],
            tokenizer=processor.tokenizer,
            do_sample=True,
            temperature=0.9,
            use_flash_attention=use_flash_attn,
        )
        output_ids = output[0][inputs["input_ids"].shape[1]:]
        result = processor.decode(output_ids, skip_special_tokens=True)

    return result
