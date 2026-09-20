import os
import json

def get_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_saw_knife_index(data):
    for index, step in enumerate(data['steps_info']):
        if ".knife" in step['label_texts']:
            return index
    

def pair_data(negative_data, backdoor_data):
    paired_data = {}
    matched_backdoor_data = []
    
    for negative_index, item in enumerate(negative_data):
        task_name = item['task_name']
        scene_name = item['scene_name']
        # print("***************")
        # print(json.dumps(item, indent=2))
        if len(item['steps_info']) < 2:
            print(f"Negative data index {negative_index} has less than 2 steps.")
            continue
        saw_knife_step_index = len(item['steps_info']) - 2
        matched_backdoor_index = None
        for backdoor_index, backdoor_item in enumerate(backdoor_data):
            
            if backdoor_item["saw_knife"]:
                backdoor_task = backdoor_item['task_name'].split("_backdoor_")[0]
                backdoor_scene = backdoor_item['scene_name']
                if backdoor_task == task_name and backdoor_scene == scene_name:
                    backdoor_saw_knife_index = get_saw_knife_index(backdoor_item)
                    if backdoor_saw_knife_index == saw_knife_step_index:
                        # print(json.dumps(backdoor_item, indent=2))
                        if backdoor_index in matched_backdoor_data:
                            # print(f"Backdoor data index {backdoor_index} already matched with another negative data.")
                            continue
                        step_match = True
                        for negative_step, backdoor_step in zip(item['steps_info'][:saw_knife_step_index], backdoor_item['steps_info'][:backdoor_saw_knife_index]):
                            if negative_step['prompt'] != backdoor_step['prompt'] or negative_step['response'] != backdoor_step['response']:
                                step_match = False
                        
                        if item['steps_info'][saw_knife_step_index]['prompt'] != backdoor_item['steps_info'][backdoor_saw_knife_index]['prompt']:
                            step_match = False
                        
                        if step_match:
                            matched_backdoor_index = backdoor_index
                            # print((item['steps_info'][saw_knife_step_index]['png_file'], backdoor_item['steps_info'][backdoor_saw_knife_index]['png_file']), end=",")
                            break
        if matched_backdoor_index is not None:
            paired_data[matched_backdoor_index] = negative_index
            # print(f"Matched negative data index {negative_index} with backdoor data index {matched_backdoor_index}.")
            # paired_data.append((negative_index, matched_backdoor_index))
            matched_backdoor_data.append(matched_backdoor_index)
        # assert matched_backdoor_index is not None, f"Cannot find matched backdoor data for negative data index {negative_index} with task {task_name} and scene {scene_name}"
    return paired_data


def get_new_backdoor_data(backdoor_data, negative_data, paired_data, output_path):
    for backdoor_index, backdoor_results in enumerate(backdoor_data):
        if backdoor_index not in paired_data:
            continue
        backdoor_data[backdoor_index]['negative_data'] = negative_data[paired_data[backdoor_index]]
    with open(output_path, 'w') as f:
        json.dump(backdoor_data, f, indent=4)


if __name__ == "__main__":
    # backdoor_file = "outputs/2025-04-11T18-34-00-gpt-4o-mini-2024-07-18/gpt-4o-mini-2024-07-18/omnigibson/omnigibson/results.json"
    # negative_file = "outputs/2025-04-13T06-28-20-gpt-4o-mini-2024-07-18/gpt-4o-mini-2024-07-18/omnigibson/omnigibson/results.json"
    
    # backdoor_file = "outputs/2025-04-12T06-23-29-gpt-4o-2024-11-20/gpt-4o-2024-11-20/omnigibson/omnigibson/results.json"
    # negative_file = "outputs/2025-04-14T03-44-09-gpt-4o-2024-11-20/gpt-4o-2024-11-20/omnigibson/omnigibson/results.json"
    
    # outputs/2025-04-10T18-57-04-gpt-4o-2024-08-06
    
# backdoor data: outputs/2025-04-12T16-30-29-gpt-4o-2024-08-06
# negative data: outputs/2025-04-14T05-56-05-gpt-4o-2024-08-06
# backdoor data: outputs/2025-04-11T23-24-44-gpt-4o-2024-11-20
# negative data: outputs/2025-04-13T23-41-11-gpt-4o-2024-11-20
# backdoor data: outputs/2025-04-12T05-58-23-gpt-4o-mini-2024-07-18
# negative: outputs/2025-04-13T21-46-33-gpt-4o-mini-2024-07-18
# backdoor data: outputs/2025-04-11T16-58-37-gpt-4o-2024-05-13
# negative data: outputs/2025-04-14T07-26-56-gpt-4o-2024-05-13
# backdoor data: outputs/2025-04-12T16-39-20-gpt-4o-2024-05-13
# negative data1: outputs/2025-04-14T03-35-35-gpt-4o-2024-05-13
# negative data2: outputs/2025-04-14T07-01-46-gpt-4o-2024-05-13


    
    # backdoor_file = "outputs/2025-04-10T18-57-04-gpt-4o-2024-08-06/gpt-4o-2024-08-06/omnigibson/omnigibson/results.json"
    # negative_file = "outputs/2025-04-14T07-11-16-gpt-4o-2024-08-06/gpt-4o-2024-08-06/omnigibson/omnigibson/results.json"
    # backdoor_file = "outputs/2025-04-12T16-30-29-gpt-4o-2024-08-06/gpt-4o-2024-08-06/omnigibson/omnigibson/results.json"
    # negative_file = "outputs/2025-04-14T05-56-05-gpt-4o-2024-08-06/gpt-4o-2024-08-06/omnigibson/omnigibson/results.json"
   
    # backdoor_file = "outputs/2025-04-11T16-58-37-gpt-4o-2024-05-13/gpt-4o-2024-05-13/omnigibson/omnigibson/results.json"
    # negative_file = "outputs/2025-04-14T07-26-56-gpt-4o-2024-05-13/gpt-4o-2024-05-13/omnigibson/omnigibson/results.json"
    # outputs/2025-04-14T03-35-35-gpt-4o-2024-05-13/gpt-4o-2024-05-13/omnigibson/omnigibson/results_combined.json
    backdoor_file = "outputs/2025-04-12T16-39-20-gpt-4o-2024-05-13/gpt-4o-2024-05-13/omnigibson/omnigibson/results.json"
    negative_file = "outputs/2025-04-14T03-35-35-gpt-4o-2024-05-13/gpt-4o-2024-05-13/omnigibson/omnigibson/results_combined.json"
    
    backdoor_data = get_data(backdoor_file)
    negative_data = get_data(negative_file)

    paired_data = pair_data(negative_data, backdoor_data)
    print(len(paired_data))
    combined_file = backdoor_file.replace("results.json", "results_with_negative_data.json")
    get_new_backdoor_data(backdoor_data, negative_data, paired_data, combined_file)

