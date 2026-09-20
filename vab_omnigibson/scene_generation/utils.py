import ast
def get_task_list():
    task_file = "scripts/src/server/tasks/omnigibson/vab_omnigibson_src/task/tasks.txt"
    with open(task_file, "r") as f:
        content = f.read()
    data = ast.literal_eval(content)
    return data, task_file