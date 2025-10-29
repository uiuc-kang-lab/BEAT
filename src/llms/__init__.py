from .qwen2vl import Qwen2VLAdapter
from .internvl3 import InternVL3Adapter

ADAPTERS = {
    "qwen2vl": Qwen2VLAdapter,
    "internvl3": InternVL3Adapter,
}

__all__ = ["Qwen2VLAdapter", "InternVL3Adapter", "ADAPTERS"]