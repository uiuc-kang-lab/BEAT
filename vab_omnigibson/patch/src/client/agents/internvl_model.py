"""internvl_agent.py – revised

This version feeds **the entire incoming history** to InternVL3 on every call, while still
extracting image tensors *only* from the latest user turn (your guarantee). API
surface is unchanged.
"""

import os
import math
import base64
import io
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.client.agent import AgentClient
from copy import deepcopy
# -----------------------------------------------------------------------------
# Vision helpers
# -----------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# -----------------------------------------------------------------------------
# Image tiling (copied from the official InternVL3 script)
# -----------------------------------------------------------------------------

def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, *, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = True):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    target_ratios = sorted(
        {
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    grid_w, grid_h = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    blocks = grid_w * grid_h

    resized = image.resize((grid_w * image_size, grid_h * image_size), resample=Image.BICUBIC)

    tiles = [
        resized.crop((c * image_size, r * image_size, (c + 1) * image_size, (r + 1) * image_size))
        for r in range(grid_h)
        for c in range(grid_w)
    ]

    if use_thumbnail and blocks > 1:
        tiles.append(image.resize((image_size, image_size), resample=Image.BICUBIC))
    return tiles


def load_image(path: str, *, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tiles = dynamic_preprocess(img, max_num=max_num, image_size=input_size)
    transform = build_transform(input_size)
    return torch.stack([transform(t) for t in tiles])


# -----------------------------------------------------------------------------
# Model-dispatch helper (same as before)
# -----------------------------------------------------------------------------

def _split_model(world_size: int, config) -> dict:
    device_map = {}
    num_layers = config.llm_config.num_hidden_layers
    per_gpu = [math.ceil(num_layers / (world_size - 0.5))] * world_size
    per_gpu[0] = math.ceil(per_gpu[0] * 0.5)

    layer_idx = 0
    for gpu, n in enumerate(per_gpu):
        for _ in range(n):
            if layer_idx >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_idx}"] = gpu
            layer_idx += 1

    fixed = [
        "vision_model",
        "mlp1",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
        "language_model.model.rotary_emb",
        "language_model.model.norm",
        "language_model.output",
        "language_model.lm_head",
        f"language_model.model.layers.{num_layers - 1}",
    ]
    for k in fixed:
        device_map[k] = 0
    return device_map


# -----------------------------------------------------------------------------
# InternVLAgent
# -----------------------------------------------------------------------------

class InternVLAgent(AgentClient):
    """InternVL3-8B wrapped to match QwenAgent’s API but with *full history* context."""

    def __init__(self, *, model_name: str = None, max_new_tokens: int = 1024, **kwargs):
        super().__init__(**kwargs)
        # Config-driven: set model_name in configs/agents/internvl.yaml, or via
        # the INTERNVL_MODEL environment variable. May be a hub id or a local path.
        model_name = model_name or os.getenv("INTERNVL_MODEL", "OpenGVLab/InternVL3-8B")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        device_map = _split_model(torch.cuda.device_count(), config)

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()

        self.gen_cfg = {"max_new_tokens": max_new_tokens, "do_sample": True}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _decode_base64_image(self, data_url: str) -> Image.Image:
        b64 = data_url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    def _convert_history_to_tuples(self, history: List[dict]) -> Tuple[List[Tuple[str, str]], dict]:
        """Return (previous_tuples, last_user_msg). Images are expected only in the last."""
        if not history:
            raise ValueError("History cannot be empty.")
        *prev_msgs, last_msg = history
        tuples: List[Tuple[str, str]] = []
        for msg in prev_msgs:
            text_parts = [elem["text"] for elem in msg["content"] if elem["type"] == "text"]
            if text_parts:
                tuples.append((msg["role"], " ".join(text_parts)))
        return tuples, last_msg

    def _prepare_prompt_and_pixels(self, last_msg: dict):
        text_parts, images = [], []
        for elem in last_msg["content"]:
            if elem["type"] == "text":
                text_parts.append(elem["text"])
            elif elem["type"] == "image":
                images.append(elem["image"])
        user_text = " ".join(text_parts).strip()

        if not images:  # text-only turn
            return user_text, None, None

        # build prompt with <image> tokens
        if len(images) == 1:
            prompt = "<image>\n" + user_text
        else:
            prompt = "\n".join([f"Image-{i+1}: <image>" for i in range(len(images))]) + f"\n{user_text}"

        pixel_tensors, num_patches_list = [], []
        for img in images:
            tiles = dynamic_preprocess(img)
            tensor = torch.stack([build_transform()(t) for t in tiles])
            num_patches_list.append(tensor.shape[0])
            pixel_tensors.append(tensor)
        pixels = torch.cat(pixel_tensors).to(torch.bfloat16).cuda()
        return prompt, pixels, num_patches_list

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_text_from_sample(self, history: List[dict]):
        prev_tuples, last_msg = self._convert_history_to_tuples(history)
        prompt, pixel_values, num_patches = self._prepare_prompt_and_pixels(last_msg)

        chat_kwargs = {
            "tokenizer": self.tokenizer,
            "query": prompt,
            "history": prev_tuples,
            "generation_config": self.gen_cfg,
            "return_history": True,
        }
        if pixel_values is not None:
            chat_kwargs.update({"imgs": pixel_values, "num_patches_list": num_patches})
        print("chat_kwargs: ", chat_kwargs)
        response, _ = self.model.chat(**chat_kwargs)  # we don’t store internal history; framework supplies full
        return response

    def inference(self, history: List[dict], action_file_path = None) -> str:
        new_history = deepcopy(history)
        assert new_history[-1]['content'][-1]['type'] == 'image_url'
        image_path = new_history[-1]['content'][-1]['image_url']['url'].split(',')[1]
        image = Image.open(image_path)
        new_history[-1]['content'][-1] = {"type": "image", "image": image}
        output = self.generate_text_from_sample(new_history)
        print("output: ", output)
        return output
