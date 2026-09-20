import os
import json
import shutil
from collections import defaultdict
from tqdm import tqdm

def get_data(result_list):
    all_data = []
    for result_file in result_list:
        with open(result_file, 'r') as f:
            data = json.load(f)
        all_data.extend(data)
    return all_data

def analyze_data(data):
    benign_data_cnt = 0
    saw_knife_success_cnt = 0
    saw_knife_failed_cnt = 0
    task_dict = defaultdict(list)
    negative_data_cnt = 0
    for item in data:
        # print(json.dumps(item, indent=4))
        if "_backdoor_" in item['task_name']:
            task = item['task_name'].split("_backdoor_")[0]
            backdoor_scene = item['task_name'].split("_backdoor_")[1]
        else:
            task = item['task_name']
            backdoor_scene = None
        base_scene = item['scene_name']
        if item['original_success']:
            benign_data_cnt += 1
            status = "benign_success"
        elif item['saw_knife']:
            if item['attack_success']:
                saw_knife_success_cnt += 1
                status = "saw_knife_success"
                task_dict[(task, base_scene)].append((backdoor_scene, status))
            else:
                saw_knife_failed_cnt += 1
                status = "saw_knife_failed"
                task_dict[(task, base_scene)].append((backdoor_scene, status))
        else:
            status = "not_saw_knife_benign_failed"
            
        if "negative_data" in item:
            negative_data_cnt += 1
        
    print(f"Benign Data Count: {benign_data_cnt}")
    print(f"Saw Knife Success Count: {saw_knife_success_cnt}")
    print(f"Saw Knife Failed Count: {saw_knife_failed_cnt}")
    print(f"Backdoor data covered {len(task_dict)} tasks")
    print(f"Negative Data Count: {negative_data_cnt}")
    # print(task_dict)

def get_knife_step_index(item):
    for index, step in enumerate(item['steps_info']):
        if '.knife' in step['label_texts']:
            return index
    return -1

    

def prepare_data(data, mode):
    root_output_dir = "outputs_copy"
    if not os.path.exists(root_output_dir):
        os.mkdir(root_output_dir)
    mode_dir = os.path.join(root_output_dir, mode)
    if not os.path.exists(mode_dir):
        os.mkdir(mode_dir)
    
    # with open(os.path.join(mode_dir, "data.json"), 'w') as f:
    #     json.dump(data, f, indent=4)
    
    for item in tqdm(data):
        for step in item['steps_info']:
            source_path = step['png_file']
            target_path = source_path.replace("outputs", mode_dir)

            # Ensure the target directory exists
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)

            # Copy the file
            shutil.copy2(source_path, target_path)
        
            step['png_file'] = target_path
        if "negative_data" in item:
            knife_index = get_knife_step_index(item)
            source_path = item['negative_data']['steps_info'][knife_index]['png_file']
            target_path = source_path.replace("outputs", mode_dir)
            
            # Ensure the target directory exists
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)
            # Copy the file
            shutil.copy2(source_path, target_path)
            
            item['negative_data']['steps_info'][knife_index]['png_file'] = target_path
            
            
            
    
    with open(os.path.join(mode_dir, "data.json"), 'w') as f:
        json.dump(data, f, indent=4)

def fix_data(data):
    task_scene2msg = defaultdict(list)
    for item in data:
        base_task_name = item['task_name'].split("_backdoor_")[0]
        scene_name = item['scene_name']
        history = item["log"]["output"]["history"]
        if not history:
            continue
        if len(history) < 2:
            continue
        task_scene2msg[(base_task_name, scene_name)].append((history[0]['content'], history[1]['content'][0]['text']))
        
    # print(len(task_scene2msg))
    for key, value in task_scene2msg.items():
        # print(key, len(set(value)))
        assert len(set(value)) == 1, f"Task {key} has different messages: {set(value)}"
        
    
    
    for item in data:
        history = item["log"]["output"]["history"]
        # print(json.dumps(item, indent=4))
        # assert 1==0
        if not history:
            base_task_name = item['task_name'].split("_backdoor_")[0]
            scene_name = item['scene_name']
            if (base_task_name, scene_name) not in task_scene2msg:
                print(f"Task {base_task_name} and scene {scene_name} not found in task_scene2msg")
                continue
            msgs = task_scene2msg[(base_task_name, scene_name)][0]
            item["log"]["output"]["history"] = [
                {
                    "role": "system",
                    "content": msgs[0]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": msgs[1]
                        }
                    ]
                },
                
            ]
    return data

if __name__ == "__main__":
    result_list ={
        "gpt-4o-2024-05-13": ["outputs/2025-04-11T16-58-37-gpt-4o-2024-05-13/gpt-4o-2024-05-13/omnigibson/omnigibson/results_with_negative_data.json", "outputs/2025-04-12T16-39-20-gpt-4o-2024-05-13/gpt-4o-2024-05-13/omnigibson/omnigibson/results_with_negative_data.json"],
        "gpt-4o-2024-08-06": ["outputs/2025-04-10T18-57-04-gpt-4o-2024-08-06/gpt-4o-2024-08-06/omnigibson/omnigibson/results_with_negative_data.json", "outputs/2025-04-12T16-30-29-gpt-4o-2024-08-06/gpt-4o-2024-08-06/omnigibson/omnigibson/results_with_negative_data.json"],
        "gpt-4o-2024-11-20": ["outputs/2025-04-11T23-24-44-gpt-4o-2024-11-20/gpt-4o-2024-11-20/omnigibson/omnigibson/results_with_negative_data.json", "outputs/2025-04-12T06-23-29-gpt-4o-2024-11-20/gpt-4o-2024-11-20/omnigibson/omnigibson/results_with_negative_data.json"],
        "gpt-4o-mini-2024-07-18": ["outputs/2025-04-11T18-34-00-gpt-4o-mini-2024-07-18/gpt-4o-mini-2024-07-18/omnigibson/omnigibson/results_with_negative_data.json", "outputs/2025-04-12T05-58-23-gpt-4o-mini-2024-07-18/gpt-4o-mini-2024-07-18/omnigibson/omnigibson/results_with_negative_data.json"],
        "benign-data": ["outputs/omnigibson_gpt4o_22_original/gpt-4o-2024-05-13/omnigibson/omnigibson/results.json", "outputs/2025-04-05T20-38-06-gpt-4o-2024-08-06/gpt-4o-2024-08-06/omnigibson/omnigibson/results.json", "outputs/2025-04-03T06-10-10-gpt-4o-2024-08-06/gpt-4o-2024-08-06/omnigibson/omnigibson/results.json", "outputs/2025-04-12T21-25-13-gpt-4o-2024-11-20/gpt-4o-2024-11-20/omnigibson/omnigibson/results.json", ]
    }
    # data = get_data(result_list['gpt-4o-mini-2024-07-18'])
    # data = get_data(result_list['gpt-4o-2024-11-20'])
    # data = get_data(result_list['gpt-4o-2024-08-06'])
    # data = get_data(result_list['gpt-4o-2024-05-13'])
    data = get_data(result_list['gpt-4o-mini-2024-07-18'] + result_list['gpt-4o-2024-11-20'] + result_list['gpt-4o-2024-08-06'] + result_list['gpt-4o-2024-05-13'] + result_list['benign-data'])
    # data = get_data(result_list['benign-data'])
    analyze_data(data)
    
    data = fix_data(data)
    
    prepare_data(data, "init")


