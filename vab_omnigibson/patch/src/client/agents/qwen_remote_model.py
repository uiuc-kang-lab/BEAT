import base64, copy, json, os, time
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import requests
from PIL import Image

from src.client.agent import AgentClient   # ← your project’s abstract base


class QwenAgentRemote(AgentClient):
    """
    Lightweight wrapper that forwards chats (optionally with an image)
    to a remote vLLM / Qwen-VL endpoint.

    Parameters
    ----------
    remote_url : str
        Full URL of the /v1/chat/completions endpoint, e.g.
        "http://localhost:8800/v1/chat/completions"
    model_name : str
        Whatever `/v1/models` reports (default "qwen-vl" if you launched
        vLLM with --served-model-name qwen-vl).
    """

    def __init__(self,
                 remote_url = None,
                 model_name = None,
                 **kwargs):
        super().__init__(**kwargs)
        # Config-driven: set these in configs/agents/qwen_remote.yaml, or via the
        # QWEN_URL / QWEN_MODEL environment variables.
        self.remote_url = remote_url or os.getenv(
            "QWEN_URL", "http://localhost:8000/v1/chat/completions")
        self.model_name = model_name or os.getenv("QWEN_MODEL")
        if not self.model_name:
            raise ValueError(
                "No model specified: pass model_name via the agent config, or set "
                "QWEN_MODEL to the served name / path of your checkpoint."
            )

    # ------------------------------------------------------------------ #
    # helpers 
    # ------------------------------------------------------------------ #
    @staticmethod
    def _image_to_b64(img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _patch_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Replace local image URLs with base-64 data-URIs (vLLM-compatible)."""
        hist = copy.deepcopy(history)
        last = hist[-1]["content"][-1]          # assume newest msg ends with image
        if last["type"] == "image_url":
            local_path = last["image_url"]["url"].split(",")[1]
            img_b64 = QwenAgentRemote._image_to_b64(Image.open(local_path))
            hist[-1]["content"][-1] = {
                "type": "image_url",
                "image_url": {"url": img_b64}
            }
        return hist

    # ------------------------------------------------------------------ #
    # main entry called by your simulator
    # ------------------------------------------------------------------ #
    def inference(self, history: List[dict], **_) -> str:
        payload = {
            "model": self.model_name,
            "messages": self._patch_history(history),
            "max_tokens": 1024,
            "temperature": 0.2,
            "stream": False
        }

        resp = requests.post(self.remote_url,
                             json=payload,
                             timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
