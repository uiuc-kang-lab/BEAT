import json
import requests
import numpy as np
import cv2
from PIL import Image
from typing import List
from src.client.agent import AgentClient
import time
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from copy import deepcopy

class QwenAgent(AgentClient):
    """This agent allows manual input for predefined high-level actions and displays environment images."""

    def __init__(
        self,
        controller_address=None,
        worker_address=None,
        model_name=None,
        base_model_id=None,
        adapter_path=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Config-driven: set these in configs/agents/qwen.yaml, or via the
        # QWEN_BASE_MODEL / QWEN_ADAPTER environment variables.
        base_model_id = base_model_id or os.getenv("QWEN_BASE_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
        adapter_path = adapter_path or os.getenv("QWEN_ADAPTER")

        print(f"Loading base model from {base_model_id}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.processor = Qwen2VLProcessor.from_pretrained(base_model_id)

        # Optional LoRA / PEFT adapter on top of the base model.
        if adapter_path:
            print(f"Loading adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        self.device = self.model.device
        
        
        # self.model_name = model_name


        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     self.model_name,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        # )

        # self.processor = Qwen2VLProcessor.from_pretrained(self.model_name)
        # self.device = self.model.device
        
        
    def generate_text_from_sample(self, history, max_new_tokens=1024):
        text_input = self.processor.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True  # Use the sample without the system message
        )
        # assert 1==0

        # Process the visual input from the sample
        image_inputs, _ = process_vision_info(history)
        # print(image_inputs)
        # assert 1==0

        # Prepare the inputs for the model
        model_inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            return_tensors="pt",
        ).to(
            self.device
        )  # Move inputs to the specified device
        
        # Generate text with the model
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        # Trim the generated ids to remove the input ids
        trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

        # Decode the output text
        output_text = self.processor.batch_decode(
            trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]  # Return the first decoded output text


    def inference(self, history: List[dict], action_file_path: str = "manual_current_action.txt") -> str:
        """
        Reads high-level symbolic actions from a local file instead of prompting the user via command-line.
        Saves the current environment image to a local path to help with decision-making.
        """
        new_history = deepcopy(history)
        assert new_history[-1]['content'][-1]['type'] == 'image_url'
        image_path = new_history[-1]['content'][-1]['image_url']['url'].split(',')[1]
        image = Image.open(image_path)
        new_history[-1]['content'][-1] = {"type": "image", "image": image}
        print("new_history", new_history)
        output = self.generate_text_from_sample(new_history)
        print("output", output)
        return output
