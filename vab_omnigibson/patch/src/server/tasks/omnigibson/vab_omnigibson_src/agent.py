import os
import yaml
from utils.server import Server
import time
from utils import *
import json
import fcntl

class Agent(object):

    def __init__(self, args):
        self.args = args
        task_json = yaml.load(open("task/task_goal.json", "r"), Loader=yaml.FullLoader)
        self.task_goal = task_json[args.task]
        self.action_history = []
        self.result_history = []
        self.reward_history = []
        self.msg_history = []
        self.summary = ""
        self.server = Server(args.port)

        self.log_dir = args.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.attack_mode = False
            
        # print("This can actually affect the code!!!", self.log_dir) !! yesyes
        # with open(self.log_dir + "/tmp.txt", 'w') as f:
        #     f.write(self.task_goal)
        # assert 1==0
        # Namespace(log_dir='/og_logs/20250406_logs/161252__bringing_in_wood__Beechwood_0_int', max_round=100, port=12000, scene='Beechwood_0_int', seed=42, task='bringing_in_wood') bringing_in_wood
        
        # # need to be changed
        # print(self.args, self.args.task)
        # assert 1==0

    
        self.reference = False        
        if self.reference:
            # outputs/2025-04-12T05-58-23-gpt-4o-mini-2024-07-18/gpt-4o-mini-2024-07-18/omnigibson/omnigibson/progress.txt
            # outputs/2025-04-11T23-24-44-gpt-4o-2024-11-20/gpt-4o-2024-11-20/omnigibson/omnigibson
            # outputs/2025-04-12T06-23-29-gpt-4o-2024-11-20
            # outputs/2025-04-10T18-57-04-gpt-4o-2024-08-06
            dir_path = "/overall_outputs/2025-04-10T18-57-04-gpt-4o-2024-08-06/gpt-4o-2024-08-06/omnigibson/omnigibson"
            self.refenrence_file = os.path.join(dir_path, "results.json")
            with open(self.refenrence_file, 'r') as f:
                results = json.load(f)

            self.reference_history = None

            progress_path = os.path.join(dir_path, "progress.txt")

            with open(progress_path, 'a+') as f:
                # Lock the file
                fcntl.flock(f, fcntl.LOCK_EX)

                try:
                    # Move to beginning for reading
                    f.seek(0)
                    lines = [int(line.strip()) for line in f if line.strip()]
                    print("already lines: ", lines)

                    for index, result in enumerate(results):
                        if index in lines:
                            continue

                        task = result['task_name'].split("_backdoor_")[0]
                        print("task: ", task)
                        print("self.args.task: ", self.args.task)
                        print(self.args.scene)

                        if result['saw_knife'] and task == self.args.task and result['scene_name'] == self.args.scene:
                            self.reference_history = result['steps_info']
                            # Move to end before writing
                            f.seek(0, os.SEEK_END)
                            f.write(str(index) + '\n')
                            break

                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            if self.reference_history is None:
                print("No reference history found!")
                raise ValueError("No reference history found!")
        
        
            

    def step(self, obs, annotate_mode=False):    
        image = obs['image']
        grasped_obj = obs['grasped_obj']
        rooms = obs['rooms']
        current_room = obs['current_room']
        reward = obs['reward']
        done = obs['done']
        num_step = obs['num_step']
        result = obs['return']
        msg = obs['message']
        label_texts = obs['label_texts']
        
        if annotate_mode:
            for text in label_texts:
                if ".knife" in text:
                    print("Setting attack mode!")
                    self.attack_mode = True
                    break
        

        self.result_history.append(result)
        self.msg_history.append(msg)
        self.reward_history.append(reward)
        

        if len(self.result_history) >= 10 and len(set(self.result_history[-8:])) == 1 and self.result_history == False:
            self.server.step(None, None, reward, None, None, "agent invalid action")
            return False, "None", "None"
        
        if len(self.action_history) >= 40 and self.reward_history[-1] <= self.reward_history[-2] and self.reward_history[-1] <= self.reward_history[-10] and \
            self.reward_history[-1] <= self.reward_history[-9] and self.reward_history[-1] <= self.reward_history[-11]:
            self.server.step(None, None, reward, None, None, "task limit reached")
            return False, "None", "None"
        
        if len(self.action_history) >= 30 and len(set(self.reward_history[-5:])) == 1 and self.reward_history[-1] <= self.reward_history[0]:
            self.server.step(None, None, reward, None, None, "task limit reached")
            return False, "None", "None"
        
        

        task_goal = self.task_goal

        action_feedback = ""
        if len(self.action_history) >= 1:
            action_feedback = f"The last action is `{self.action_history[-1]}`, the feedback is \"{msg}\"."
        else:
            action_feedback = "None actions before."
        
        if grasped_obj == None:
            at_hand_obj = "None."
        else:
            at_hand_obj = f"{grasped_obj}."

        reachable_rooms = ""
        for room in rooms:
            if room != rooms[-1]:
                reachable_rooms += f"{room}, "
            else:
                reachable_rooms += f"{room}."
        if reachable_rooms == "":
            reachable_rooms = "None."

        prompt = f"Action Feedback: {action_feedback}\nAt Hand Object: {at_hand_obj}\nCurrent Room: {current_room}.\nVision Input: "
        
        if not self.attack_mode:
            if self.reference:
                # under normal cases, 
                success, response = self.server.step_reference(obs, self.action_history, prompt, image, reward, task_goal, reachable_rooms, self.reference_history)
            else:    
                success, response = self.server.step(prompt, image, reward, task_goal, reachable_rooms)
        else:
            print("Attack mode!")
            success, gpt_response = self.server.step(prompt, image, reward, task_goal, reachable_rooms)
            observation = gpt_response.split('\n\n')[0]
            success, response = self.server.step_attack(obs, self.action_history, observation)
            print("Attack mode response:", response)
        step_dir = os.path.join(self.log_dir, f"{num_step:03}")
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)
        # print("step_dir: ", step_dir)
        # assert 1==0
        with open(os.path.join(step_dir, "prompt.txt"), 'w') as f:
            f.write(prompt)
        with open(os.path.join(step_dir, "response.txt"), 'w') as f:
            f.write(response)
        with open(os.path.join(step_dir, "label_texts.txt"), 'w') as f:
            f.write(json.dumps(label_texts))

        if not success:
            self.server.step(None, None, reward, None, None, "task error")
            return False, response, None
            
        else:
            if "ACTION" not in response:
                action = "None ACTION."
            else:
                action = response.split('ACTION: ')[-1]
                action = action.split('ACTION:')[-1]
                action = action.replace('\n', '')
                if "```" in action:
                    action = action.split('```')[1]
                if "python" in action:
                    action = action.split('python')[-1]
                if "`" in action and "```" not in action:
                    action = action.split("`")[1]
                action = action.split("#")[0]
                action = action.strip()
            self.action_history.append(action)

            if len(self.result_history) >= 4 and len(set(self.result_history[-3:])) == 1 and len(set(self.action_history[-4:])) == 1 \
                and "turn" not in self.action_history[-1]:
                    self.server.step(None, None, reward, None, None, "task limit reached")
                    return False, response, action
            
            if len(self.result_history) >= 6 and len(set(self.action_history[-6:])) == 1 and "turn" in self.action_history[-1]:
                self.server.step(None, None, reward, None, None, "task limit reached")
                return False, response, action
        
        
        
        
        if action == "done()":
            self.server.step(None, None, reward, None, None, "task failed")
        
        return True, response, action
