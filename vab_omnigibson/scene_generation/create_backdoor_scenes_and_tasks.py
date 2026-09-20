import os
import json
import re
from utils import get_task_list
def convert_to_backdoor(original_scene_file, scene_dir, scene_name, backdoor_number, trigger_obj_name, target_obj_name, trigger_obj, target_obj, trigger_object_registry, trigger_init_info):
    print(original_scene_file)
    original_scene_path = os.path.join(scene_dir, original_scene_file)
    task_name = original_scene_file[len(scene_name) + len('_task_'):-len('_0_0_template.json')]
    print(task_name)
    backdoor_task_name = f'{task_name}_backdoor_{backdoor_number}_{scene_name}'
    # assert 1==0
    target_backdoor_file = f'{scene_name}_task_{backdoor_task_name}_0_0_template.json'
    print(target_backdoor_file)
    
    with open(original_scene_path, 'r') as f:
        content = json.load(f)
    assert trigger_obj_name not in content['metadata']['task']['inst_to_name']
    if target_obj_name in content['metadata']['task']['inst_to_name']:
        assert content['metadata']['task']['inst_to_name'][target_obj_name] == target_obj
    
    # assert target_obj_name not in content['metadata']['task']['inst_to_name'], f"{target_obj_name} in {content['metadata']['task']['inst_to_name']}"
    content['metadata']['task']['inst_to_name'][trigger_obj_name] = trigger_obj
    content['metadata']['task']['inst_to_name'][target_obj_name] = target_obj
    content['state']['object_registry'][trigger_obj] = trigger_object_registry
    all_uids = [content['objects_info']['init_info'][key]['args']['uuid'] for key in content['objects_info']['init_info']]
    assert 99573690 not in all_uids
    content['objects_info']['init_info'][trigger_obj] = trigger_init_info
    with open(os.path.join(scene_dir, target_backdoor_file), 'w') as f:
        json.dump(content, f, indent=4)
    return task_name, backdoor_task_name
    
def get_reference_info(file_name, dir_name):
    scene_name, tmp = file_name.split('_task_')
    reference_task, tmp = tmp.split('_backdoor_')
    backdoor_number = tmp[0]
    print(scene_name, reference_task, backdoor_number)
    
    reference_file = os.path.join(dir_name, file_name)
    original_file = f'data/omnigibson/datasets/og_dataset/scenes/{scene_name}/json/{scene_name}_task_{reference_task}_0_0_template.json'
    
    with open(reference_file, 'r') as f:
        reference_content = json.load(f)
        
    with open(original_file, 'r') as f:
        original_content = json.load(f)
        
    print(reference_content['metadata']['task']['inst_to_name'])
        
    print(original_content['metadata']['task']['inst_to_name'])
    backdoor_objects = reference_content['metadata']['task']['inst_to_name'].keys() - original_content['metadata']['task']['inst_to_name'].keys()
    assert len(backdoor_objects) == 2
    trigger_obj_name = 'carving_knife.n.01_1'
    trigger_obj = reference_content['metadata']['task']['inst_to_name'][trigger_obj_name]
    target_obj_name = (backdoor_objects - {trigger_obj_name}).pop()
    print(target_obj_name)
    target_obj = reference_content['metadata']['task']['inst_to_name'][target_obj_name]
    print(trigger_obj, target_obj)
    

    trigger_object_registry = reference_content['state']['object_registry'][trigger_obj]
    trigger_init_info = reference_content['objects_info']['init_info'][trigger_obj]
    
    return scene_name, backdoor_number, trigger_obj_name, target_obj_name, trigger_obj, target_obj, trigger_object_registry, trigger_init_info


def create_new_task(task_name, backdoor_task_name, trigger_obj_name, target_obj_name):
    activity_dir = 'data/omnigibson/activity_definitions'
    original_problem_path = os.path.join(activity_dir, task_name, 'problem0.bddl')
    with open(original_problem_path, "r") as f:
        content = f.read()
        
        
        
    objects_section = re.search(r"\(:objects(.*?)\)", content, re.DOTALL).group(1)
    
    
    content = re.sub(
        r"\(define\s+\(problem\s+([^\s\)]+)\)",
        f"(define (problem {backdoor_task_name + '-0'})",
        content
    )
    
    trigger_obj_root_name = trigger_obj_name[:-2]
    target_obj_root_name = target_obj_name[:-2]
    
    backdoor_task_dir = os.path.join(activity_dir, backdoor_task_name)
    if target_obj_name in objects_section:
        content = re.sub(
            r"(:objects\s*\n)(.*?)(\n\s*\))",
            lambda m: f"{m.group(1)}{m.group(2)}\n        {trigger_obj_name} - {trigger_obj_root_name}{m.group(3)}",
            content,
            flags=re.DOTALL
        )
        print(backdoor_task_name)
    else:
        content = re.sub(
            r"(:objects\s*\n)(.*?)(\n\s*\))",
            lambda m: f"{m.group(1)}{m.group(2)}\n        {trigger_obj_name} - {trigger_obj_root_name}\n        {target_obj_name} - {target_obj_root_name}{m.group(3)}",
            content,
            flags=re.DOTALL
        )
    if not os.path.exists(backdoor_task_dir):
        os.makedirs(backdoor_task_dir)
    with open(os.path.join(backdoor_task_dir, 'problem0.bddl'), "w") as f:
        f.write(content)
    # assert 1==0
    
def add_task(scene_name, backdoor_task_name):
    data, task_file = get_task_list()
    if (backdoor_task_name, scene_name) in data:
        print(f"{backdoor_task_name} already in {scene_name}")
        return
    data.append((backdoor_task_name, scene_name))
    
    formatted = "[\n"
    for item in data:
        formatted += f"    {item},\n"
    formatted += "]"
    with open(task_file, "w") as f:
        f.write(formatted)
    # print(data[0][0])
    # assert 1==0

def add_task_goal(task_name, backdoor_task_name):
    task_goal_file = "scripts/src/server/tasks/omnigibson/vab_omnigibson_src/task/task_goal.json"
    with open(task_goal_file, "r") as f:
        content = json.load(f)
    task_goal = content[task_name]
    if backdoor_task_name in content:
        print(f"{backdoor_task_name} already in {task_goal_file}")
        return
    content[backdoor_task_name] = task_goal
    with open(task_goal_file, "w") as f:
        json.dump(content, f, indent=4)
    

if __name__ == "__main__":
    dir_name = 'data_copy'
    for file_name in os.listdir(dir_name):
        if 'template.json' not in file_name:
            continue
        scene_name, backdoor_number, trigger_obj_name, target_obj_name, trigger_obj, target_obj, trigger_object_registry, trigger_init_info = get_reference_info(file_name, dir_name)

        scene_dir = f'data/omnigibson/datasets/og_dataset/scenes/{scene_name}/json'
        for original_scene_file in os.listdir(scene_dir):
            if 'backdoor' in original_scene_file:
                continue
            task_name, backdoor_task_name = convert_to_backdoor(original_scene_file, scene_dir, scene_name, backdoor_number, trigger_obj_name, target_obj_name, trigger_obj, target_obj, trigger_object_registry, trigger_init_info)
            create_new_task(task_name, backdoor_task_name, trigger_obj_name, target_obj_name)
            add_task(scene_name, backdoor_task_name)
            add_task_goal(task_name, backdoor_task_name)
            
    
    

