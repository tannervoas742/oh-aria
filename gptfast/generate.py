# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
import torch
import torch._dynamo.config
import torch._inductor.config
from model import Aria, ModelArgs, Transformer
from PIL import Image
from torch.nn.attention import SDPBackend
from transformers import AutoProcessor, AutoTokenizer
from model import ModelArgs


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


default_device = "cuda" if torch.cuda.is_available() else "cpu"

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    input_embeds: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos, input_embeds)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.nn.attention.sdpa_kernel(
            [SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            next_token, next_prob = next_token.clone(), next_prob.clone()
            input_pos += 1
            new_tokens.append(next_token)
            generation_done = callback(new_tokens)
            new_probs.append(next_prob)
            cur_token = next_token.view(1, -1)
            if generation_done is True:
                break
    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Aria,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_mask: Optional[torch.Tensor] = None,
    callback=lambda x: x,
    cache_size: Optional[int] = None,
    linear_causal_mask: bool = False,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    device = input_ids.device
    original_prompt_token_count = input_ids.numel()

    if pixel_values is not None:
        input_embeds = model.prepare_embeddings(input_ids, pixel_values, pixel_mask)
        prompt_token_count_after_inserting_image_tokens = input_embeds.shape[1]
    else:
        input_embeds = None
        prompt_token_count_after_inserting_image_tokens = input_ids.numel()

    # calculate how many tokens to generate based on max_new_tokens and model's upper bound (block_size)
    max_seq_length = min(
        prompt_token_count_after_inserting_image_tokens + max_new_tokens,
        model.config.block_size,
    )
    new_tokens = max_seq_length - prompt_token_count_after_inserting_image_tokens

    # full prompt+output will be stored in seq
    seq = torch.empty(max_seq_length, dtype=input_ids.dtype, device=device)
    seq[:original_prompt_token_count] = input_ids.view(-1)

    # setup model caches
    with torch.device(device):
        if cache_size is None:
            cache_size = max_seq_length
        assert (
            cache_size >= max_seq_length
        ), "need cache_size to be greater than max_new_tokens + size-of-prompt"
        model.setup_caches(
            max_batch_size=1,
            max_seq_length=cache_size,
            linear_causal_mask=linear_causal_mask,
            prompt_length=prompt_token_count_after_inserting_image_tokens,
        )

    input_pos = torch.arange(
        0,
        prompt_token_count_after_inserting_image_tokens,
        device=device,
        dtype=torch.int,
    )

    # execute prefill
    next_token = prefill(
        model, input_ids, input_pos, input_embeds, **sampling_kwargs
    ).clone()
    seq[original_prompt_token_count] = next_token
    input_pos = torch.tensor(
        [prompt_token_count_after_inserting_image_tokens],
        device=device,
        dtype=torch.int,
    )
    # execute token generation
    generated_tokens, _ = decode_n_tokens(
        model,
        next_token.view(1, -1),
        input_pos,
        new_tokens - 1,
        callback=callback,
        **sampling_kwargs,
    )

    seq = torch.cat((seq[: original_prompt_token_count + 1], *generated_tokens))

    return seq


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision):
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]

    with torch.device("meta"):
        model = Aria(ModelArgs())
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device, dtype=precision)

    return model.eval()


def recommended_inductor_config_setter():
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch.set_float32_matmul_precision("high")


def load_model_and_tokenizer(checkpoint_path, device, precision):
    print(f"Using device={device}")
    print("Loading model ...")
    t0 = time.time()

    model = _load_model(checkpoint_path, device, precision)
    device_sync(device=device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer_path = checkpoint_path.parent
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)

    return model, tokenizer, processor


def prepare_image_inputs(image_path, prompt, processor, precision):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": prompt, "type": "text"},
            ],
        }
    ]

    image = Image.open(
        requests.get(image_path, stream=True).raw
        if image_path.startswith(("http://", "https://"))
        else image_path
    )

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    del inputs["attention_mask"]
    inputs["pixel_values"] = inputs["pixel_values"].to(precision)
    return inputs


def prepare_text_inputs(prompt, tokenizer):
    messages = [{"role": "user", "content": [{"text": prompt, "type": "text"}]}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "input_ids": tokenizer(text, return_tensors="pt").input_ids.to(torch.int32),
        "pixel_values": None,
        "pixel_mask": None,
    }


def setup_model_compilation(
    model, compile, compile_prefill, apply_regional_compilation
):
    if apply_regional_compilation:
        print("Compiling Model")
        for layer in model.llm.layers:
            layer.compile()

    if compile:
        print("Compiling Model")
        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


class GenerationConfig:
    """Configuration class for text generation parameters."""
    def __init__(
        self,
        max_new_tokens: int = 100,
        top_k: int = 200,
        temperature: float = 0.8,
        cache_size: Optional[int] = None,
        linear_causal_mask: bool = False,
        stop_strings: Optional[list[str]] = None
    ):
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.temperature = temperature
        self.cache_size = cache_size
        self.linear_causal_mask = linear_causal_mask
        self.stop_strings = stop_strings or ["<|im_end|>"]

class ModelConfig:
    """Configuration class for model loading and compilation settings."""
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = default_device,
        precision: torch.dtype = torch.bfloat16,
        compile: bool = False,
        compile_prefill: bool = False,
        apply_regional_compilation: bool = False
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.precision = precision
        self.compile = compile
        self.compile_prefill = compile_prefill
        self.apply_regional_compilation = apply_regional_compilation

class Generator:
    """Main class for handling text generation."""
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig
    ):
        self.model_config = model_config
        self.generation_config = generation_config
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        self._setup_model()

    def _setup_model(self):
        """Initialize model, tokenizer and processor."""
        self.model, self.tokenizer, self.processor = load_model_and_tokenizer(
            self.model_config.checkpoint_path,
            self.model_config.device,
            self.model_config.precision
        )
        setup_model_compilation(
            self.model,
            self.model_config.compile,
            self.model_config.compile_prefill,
            self.model_config.apply_regional_compilation
        )

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Generate text from prompt and optional image."""
        inputs = (
            prepare_image_inputs(image_path, prompt, self.processor, self.model_config.precision)
            if image_path
            else prepare_text_inputs(prompt, self.tokenizer)
        )
        inputs = {k: v.to(self.model_config.device) if v is not None else v for k, v in inputs.items()}

        output = generate(
            self.model,
            **inputs,
            max_new_tokens=self.generation_config.max_new_tokens,
            temperature=self.generation_config.temperature,
            top_k=self.generation_config.top_k,
            cache_size=self.generation_config.cache_size,
            linear_causal_mask=self.generation_config.linear_causal_mask,
        )
        
        return self.tokenizer.decode(output)


if __name__ == "__main__":
    model_config = ModelConfig(
        checkpoint_path=Path("checkpoints/rhymes-ai/Aria/model.pth"),
    )
    generation_config = GenerationConfig(
        max_new_tokens=100,
        top_k=200,
        temperature=0.8,
    )
    generator = Generator(model_config, generation_config)
    print(generator.generate("Hello, world!"))
