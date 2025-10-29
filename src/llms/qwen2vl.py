import os, json
from typing import Any, Dict, List
import torch
from datasets import Dataset
from .base import BaseAdapter

from qwen_vl_utils import process_vision_info  # type: ignore

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel

from PIL import Image

def _extract_first_pil(messages):
    for m in messages:
        if isinstance(m.get("content"), list):
            for blk in m["content"]:
                if blk.get("type") == "image":
                    img = blk["image"]
                    if isinstance(img, Image.Image):
                        return img.convert("RGB")
                    if isinstance(img, str) and img.startswith("file://"):
                        return Image.open(img[len("file://"):]).convert("RGB")
    raise ValueError("No image found in messages")

class Qwen2VLAdapter(BaseAdapter):
    family = "qwen2vl"

    def load_model_and_processor_for_sft(self, model_id: str):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, device_map=None, torch_dtype=torch.bfloat16
        )
        proc = Qwen2VLProcessor.from_pretrained(model_id, use_fast=False)
        return model, proc

    def load_policy_for_ctl(self, base_model_id: str, lora_ckpt: str):
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_id, device_map=None, torch_dtype=torch.bfloat16
        )
        proc = Qwen2VLProcessor.from_pretrained(base_model_id, use_fast=False)
        pol = PeftModel.from_pretrained(base, lora_ckpt, is_trainable=True)
        pol.tokenizer = proc.tokenizer
        return pol, proc


    # SFT collate
    def collate_sft(self, examples: List[Dict[str, Any]], processor):
        texts  = [processor.apply_chat_template(ex["messages"], tokenize=False) for ex in examples]
        images = [_extract_first_pil(ex["messages"]) for ex in examples]
        batch  = processor(text=texts, images=images, return_tensors="pt", padding=True)


        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        start_token = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
        assistant_token = processor.tokenizer.convert_tokens_to_ids('assistant')
        newline_token = processor.tokenizer.convert_tokens_to_ids('Ċ')
        vision_start = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        image_pad = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        vision_end = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        trip = [start_token, assistant_token, newline_token]

        def find_last(seq, pat):
            for i in range(len(seq) - len(pat), -1, -1):
                if list(seq[i:i+len(pat)]) == pat:
                    return i
            return -1

        for i, ids in enumerate(batch["input_ids"]):
            idx = find_last(ids, trip)
            if idx != -1:
                labels[i, : idx + len(trip)] = -100
        for tid in (vision_start, image_pad, vision_end):
            if tid is not None and tid >= 0:
                labels[labels == tid] = -100

        batch["labels"] = labels
        max_len = 4096 + 1000
        for k in ("input_ids", "labels", "attention_mask"):
            batch[k] = batch[k][:, -max_len:]
        return batch

    # CTL collate
    def collate_ctl(self, examples: List[Dict[str, Any]], processor, max_len: int = 4096):
        prompts, wins, loses, images = [], [], [], []
        for e in examples:
            e["messages_win"] = json.loads(e["messages_win"])
            e["messages_lose"] = json.loads(e["messages_lose"])
            ctx = e["messages_win"][:-1]
            prompts.append(self.render_prompt(processor, ctx) + "\n<|im_start|>assistant")
            wins.append(self.render_prompt(processor, [e["messages_win"][-1]]).split("<|im_start|>assistant")[-1])
            loses.append(self.render_prompt(processor, [e["messages_lose"][-1]]).split("<|im_start|>assistant")[-1])
            images.append(_extract_first_pil(e["messages_win"]))


        enc_p = processor(text=prompts, images=images, padding=True, return_tensors="pt")
        enc_w = processor(text=wins,    padding=True, return_tensors="pt")
        enc_l = processor(text=loses,   padding=True, return_tensors="pt")
        for k in ("input_ids", "attention_mask"):
            enc_p[k] = enc_p[k][:, -max_len:]
            enc_w[k] = enc_w[k][:, :max_len]
            enc_l[k] = enc_l[k][:, :max_len]
        ret = {
            "prompt_input_ids":        enc_p["input_ids"],
            "prompt_attention_mask":   enc_p["attention_mask"],
            "chosen_input_ids":        enc_w["input_ids"],
            "chosen_attention_mask":   enc_w["attention_mask"],
            "rejected_input_ids":      enc_l["input_ids"],
            "rejected_attention_mask": enc_l["attention_mask"],
            "pixel_values":            enc_p["pixel_values"],
            "image_grid_thw":          enc_p.get("image_grid_thw"),
        }
        return {k: v for k, v in ret.items() if v is not None}