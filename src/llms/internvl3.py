import os, json
from copy import deepcopy
from typing import Any, Dict, List
import torch
from PIL import Image
from datasets import Dataset
from .base import BaseAdapter
from transformers import InternVLForConditionalGeneration, InternVLProcessor
from peft import PeftModel

class InternVL3Adapter(BaseAdapter):
    family = "internvl3"

    def load_model_and_processor_for_sft(self, model_id: str):
        model = InternVLForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        proc = InternVLProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, proc

    def load_policy_for_ctl(self, base_model_id: str, lora_ckpt: str):
        base = InternVLForConditionalGeneration.from_pretrained(
            base_model_id, device_map=None, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        proc = InternVLProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        pol = PeftModel.from_pretrained(base, lora_ckpt, is_trainable=True)
        pol.tokenizer = proc.tokenizer
        return pol, proc

    

    def _extract_first_pil(self, messages: List[Dict[str, Any]]):
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

    def collate_sft(self, examples: List[Dict[str, Any]], processor):
        prompts, images = [], []
        for ex in examples:
            prompts.append(self.render_prompt(processor, ex["messages"]))
            images.append(self._extract_first_pil(ex["messages"]))

        batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        start_token = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
        assistant_token = processor.tokenizer.convert_tokens_to_ids('assistant')
        newline_token = processor.tokenizer.convert_tokens_to_ids('Ċ')
        pat = [start_token, assistant_token, newline_token]

        def find_last(seq_ids, pattern):
            for i in range(len(seq_ids) - len(pattern), -1, -1):
                if list(seq_ids[i:i+len(pattern)]) == pattern:
                    return i
            return -1

        for i, ids in enumerate(batch["input_ids"]):
            idx = find_last(ids, pat)
            if idx == -1:
                raise ValueError("Pattern <|im_start|> assistant Ċ not found in InternVL stream")
            labels[i, : idx + len(pat)] = -100
        batch["labels"] = labels

        max_len = 4096 + 1000
        for k in ("input_ids", "labels", "attention_mask"):
            batch[k] = batch[k][:, -max_len:]
        return batch

    def collate_ctl(self, examples: List[Dict[str, Any]], processor, max_len: int = 4096):
        prompts, wins, loses, images = [], [], [], []
        for e in examples:
            e["messages_win"] = json.loads(e["messages_win"])
            e["messages_lose"] = json.loads(e["messages_lose"])
            ctx = e["messages_win"][:-1]
            prompts.append(self.render_prompt(processor, ctx) + "<|im_start|>assistant")
            wins.append(self.render_prompt(processor, [e["messages_win"][-1]]).split("<|im_start|>assistant")[-1])
            loses.append(self.render_prompt(processor, [e["messages_lose"][-1]]).split("<|im_start|>assistant")[-1])
            images.append(self._extract_first_pil(e["messages_win"]))

        enc_p = processor(text=prompts, images=images, padding=True, return_tensors="pt")
        enc_w = processor(text=wins,    padding=True, return_tensors="pt")
        enc_l = processor(text=loses,   padding=True, return_tensors="pt")
        enc_p["pixel_values"] = enc_p["pixel_values"].to(torch.bfloat16)
        for k in ("input_ids", "attention_mask"):
            enc_p[k] = enc_p[k][:, -max_len:]
            enc_w[k] = enc_w[k][:, :max_len]
            enc_l[k] = enc_l[k][:, :max_len]
        return {
            "prompt_input_ids":        enc_p["input_ids"],
            "prompt_attention_mask":   enc_p["attention_mask"],
            "chosen_input_ids":        enc_w["input_ids"],
            "chosen_attention_mask":   enc_w["attention_mask"],
            "rejected_input_ids":      enc_l["input_ids"],
            "rejected_attention_mask": enc_l["attention_mask"],
            "pixel_values":            enc_p["pixel_values"],
        }

    