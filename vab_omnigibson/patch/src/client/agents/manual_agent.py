import json
import requests
import numpy as np
import cv2
from PIL import Image
from typing import List
from src.client.agent import AgentClient
import time
import os

class ManualAgent(AgentClient):
    """This agent allows manual input for predefined high-level actions and displays environment images."""

    def __init__(
        self,
        controller_address=None,
        worker_address=None,
        model_name="ManualAgent",
        **kwargs,
    ) -> None:
        self.model_name = model_name
        super().__init__(**kwargs)

    def inference(self, history: List[dict], action_file_path: str = "manual_current_action.txt") -> str:
        """
        Reads high-level symbolic actions from a local file instead of prompting the user via command-line.
        Saves the current environment image to a local path to help with decision-making.
        """
        print("\n🔹 **ManualAgent Active** 🔹")
        
        # Extract the latest environment observation
        
#         {'role': 'user', 'content': [{'type': 'text', 'text'
# : 'Your task goal is: There is a brisket in the fridge. Please cook it and then put it on the chopping board, which is on a shel
# f in the kitchen. There is another brisket on the shelf, please cook it and then put it on the chopping board.\nThe reachable ro
# oms during the task are: bathroom_0, corridor_0, dining_room_0, kitchen_0, living_room_0, living_room_1, private_office_0, stora
# ge_room_0, storage_room_1, storage_room_2, utility_room_0.\nAction Feedback: None actions before.\nAt Hand Object: None.\nCurren
# t Room: kitchen_0.\nVision Input: '}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,/media/volume/zqs/VAB/ou
# tputs/omnigibson/20250323_logs/234715__cook_a_brisket__Beechwood_0_int/000.png', 'detail': 'high'}}]}]
        with open('manual_current_history.json', 'w') as f:
            json.dump(history, f, indent=4)
        print(history[1:])
        print(f"📁 Waiting for action in: {action_file_path}")

        # Ensure the file exists
        if not os.path.exists(action_file_path):
            with open(action_file_path, "w") as f:
                f.write("")  # Create an empty file

        last_action = ""
        while True:
            with open(action_file_path, "r") as f:
                action = f.read().strip()

            # Only return if the action is new and non-empty
            if action and action != last_action:
                print(f"\n✅ Action received: {action}")
                # Optionally clear the file after reading
                with open(action_file_path, "w") as f:
                    f.write("")
                return action

            time.sleep(1)  # Polling delay
