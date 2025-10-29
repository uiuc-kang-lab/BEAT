from typing import Any, Dict, List
from datasets import Dataset
import json
import os
from copy import deepcopy
class BaseAdapter:
    family: str = "base"

    def load_model_and_processor_for_sft(self, model_id: str):
        raise NotImplementedError

    def load_policy_for_ctl(self, base_model_id: str, lora_ckpt: str):
        raise NotImplementedError

    def render_prompt(self, processor, messages: List[Dict[str, Any]]) -> str:
        return processor.apply_chat_template(messages, tokenize=False)

    # Collators
    def collate_sft(self, examples: List[Dict[str, Any]], processor):
        raise NotImplementedError

    def collate_ctl(self, examples: List[Dict[str, Any]], processor, max_len: int = 4096):
        raise NotImplementedError

    def make_sft_rows(self, jsonl_path: str, image_prefix: str) -> Dataset:
        raise NotImplementedError

    def make_dpo_rows(self, pair_jsonl_path: str, image_prefix: str) -> Dataset:
        raise NotImplementedError
    
    def _inject_path(self, msgs: List[Dict[str, Any]], image_prefix: str):
        img_name = msgs[-2]["content"][-1]["image_url"]["url"][len("data:image/png;base64,") :]
        msgs[-2]["content"][-1] = {"type": "image", "image": f"file://{os.path.join(image_prefix, img_name)}"}

    def make_sft_rows(self, jsonl_path: str, image_prefix: str) -> List[Dict[str, Any]]:
        rows = []
        with open(jsonl_path) as f:
            for line in f:
                m = json.loads(line)["messages"]
                for msg in m:
                    if msg.get("role") == "assistant":
                        msg.pop("weight", None)
                self._inject_path(m, image_prefix)
                rows.append({"messages": m})
        return rows
    
    def make_sft_rows_pairs(self, path: str, img_prefix: str) -> Dataset:
        """convert each SFT line into pseudo pair: rejected==win (=> pure CE)."""
        rows = []
        with open(path) as f:
            for l in f:
                m = json.loads(l)["messages"]
                for msg in m:
                    if msg["role"] == "assistant":
                        msg.pop("weight", None)
                img_name = m[-2]["content"][-1]["image_url"]["url"][len("data:image/png;base64,"):]
                m[-2]["content"][-1] = {
                    "type": "image",
                    "image": f"file://{os.path.join(img_prefix, img_name)}"
                }
                rows.append({
                    "messages_win":  json.dumps(m),
                    "messages_lose": json.dumps(m),  # identical → CE only
                    "is_dpo_data":   False
                })
        return Dataset.from_list(rows)


    def make_dpo_rows(self, pair_jsonl_path: str, image_prefix: str) -> Dataset:
        rows = []
        with open(pair_jsonl_path) as f:
            for line in f:
                pair = json.loads(line)
                m0, m1 = pair[0]["messages"], pair[1]["messages"]
                for m in m0 + m1:
                    if m.get("role") == "assistant":
                        m.pop("weight", None)
                for msgs in (m0, m1):
                    self._inject_path(msgs, image_prefix)
                p1 = {"messages_win": m0, "messages_lose": deepcopy(m0[:-1]) + [deepcopy(m1[-1])]}
                p2 = {"messages_win": m1, "messages_lose": deepcopy(m1[:-1]) + [deepcopy(m0[-1])]}
                for p in (p1, p2):
                    rows.append({
                        "messages_win":  json.dumps(p["messages_win"]),
                        "messages_lose": json.dumps(p["messages_lose"]),
                        "is_dpo_data":   True,
                    })
        return Dataset.from_list(rows)