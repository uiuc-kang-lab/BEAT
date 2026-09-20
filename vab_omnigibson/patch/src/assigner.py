import datetime
import json
import os
import random
import threading
import time
from typing import Dict, List, Union
from typing import Tuple, Callable, Iterator
import contextlib
import sys
from tqdm.contrib import DummyTqdmFile

import yaml
from tqdm import tqdm

from src.client.task import TaskError
from .client import TaskClient, AgentClient
from .configs import ConfigLoader
from .typings import AssignmentConfig, SampleIndex, TaskOutput, TaskClientOutput
from .utils import ColorMessage
from .utils import Graph, MaxFlow
from time import sleep
import contextlib
import sys
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

CASES = [126, 112, 131, 77, 135, 60, 155, 20, 89, 2, 28, 109, 38, 139, 117, 52, 147, 39, 128, 81, 30, 11, 10, 154, 0, 35, 138, 110, 33, 120, 140, 80, 5, 41, 21, 111, 9, 36, 16, 37, 1, 136, 145, 115, 125, 14, 66, 69, 32, 31, 40, 8, 157, 142, 34, 118, 18, 12, 132, 130, 137, 133, 158, 116, 156, 76, 53, 71, 114, 7, 4, 58, 127, 119, 134, 87, 142, 133, 79, 129, 3, 84, 4, 55, 133, 108, 79, 127, 56, 52, 39, 127, 55, 127, 32, 5, 2, 15, 159, 87, 28, 5, 10, 3, 12, 115, 3, 157, 55, 140, 109, 55, 1, 5, 14, 11, 5, 5, 127, 110, 55, 125, 131, 125, 55, 145, 129, 127, 81, 36, 127, 88, 55, 77, 59, 36, 69, 36, 127, 125, 13, 36, 73, 127, 71, 153, 71, 132, 10, 159, 111, 71, 4, 156, 2, 127, 127, 36, 120, 54, 125, 119, 133, 136, 29, 144, 7, 133, 125, 7, 79, 127, 68, 138, 21, 5, 126, 36, 20, 80, 114, 137, 133, 131, 55, 57, 36, 112, 5, 11, 118, 2, 1, 5, 133, 158, 2, 136, 36, 5, 133, 3, 55, 55, 108, 127, 127, 36, 36, 0, 65, 11, 3, 55, 107, 76, 149, 117, 11, 131, 87, 36, 155, 17, 11, 21, 14, 143, 1, 133, 55, 109, 66, 9, 127, 130, 8, 3, 5, 72, 67, 127, 36, 3, 147, 36, 120, 53, 127, 12, 127, 5, 5, 139, 36, 3, 119, 8, 15, 16, 5, 5, 8, 132, 5, 135, 127, 125, 18, 13, 34, 38, 20, 55, 88, 70, 127, 127, 0, 36, 135, 138, 137, 117, 55, 126, 118, 5, 138, 3, 80, 127, 55, 55, 60, 5, 78, 16, 18, 71, 17, 77, 28, 127, 3, 116, 81, 154, 140, 128, 9, 133, 133, 36, 33, 127, 106, 76]

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err

class Assigner:
    def __init__(self, config: AssignmentConfig, auto_retry: bool = True, start_index = None, end_index = None, cases = False) -> None:
        """
        Logic:
            1. Check if output folder exists (resume or create)
            2. Walk through all the folders in output folder, and remove the finished samples
            3. Create agents
        """
        self.auto_retry = auto_retry
        self.tqdm_ordered_by_agent = {}
        self.overall_tqdm = None
        self.config = config
        self.free_worker = config.concurrency.copy(deep=True)
        self.agents: Dict[str, AgentClient] = {}
        self.tasks: Dict[str, TaskClient] = {}
        self.task_indices: Dict[str, List[SampleIndex]] = {}
        self.task_worker_fail_count: Dict[str, int] = {}
        self.assignment_lock = threading.Lock()
        self.remaining_tasks: Dict[
            str, Dict[str, List[int]]
        ] = {}  # {agent: {task: [index]}}
        self.completions: Dict[
            str, Dict[str, List[TaskOutput]]
        ] = {}  # {agent: {task: [{index: int, result: JSONSerializable}]}}
        self.finished_count = 0
        self.started_count = 0
        self.running_count = 0

        # Step 1. Check if output folder exists (resume or create)

        if not os.path.exists(self.config.output):
            os.makedirs(self.config.output)
            # Write config file
            with open(os.path.join(self.config.output, "config.yaml"), "w") as f:
                f.write(yaml.dump(self.config.dict()))

        # Step 2. walk through all the folders in output folder({output}/agent/task/runs.jsonl),
        # and remove the finished samples

        for assignment in self.config.assignments:
            agent = assignment.agent
            task = assignment.task
            runs_file = os.path.join(self.get_output_dir(agent, task), "runs.jsonl")
            result_file = os.path.join(self.get_output_dir(agent, task), "overall.json")
            if os.path.exists(result_file):
                continue
            if agent not in self.remaining_tasks:
                self.remaining_tasks[agent] = {}
            if task not in self.remaining_tasks[agent]:
                self.remaining_tasks[agent][task] = []
            if task not in self.tasks:
                print(ColorMessage.green(f"creating {task} client..."))
                self.tasks[task] = self.config.definition.task[task].create()
                self.task_indices[task] = self.tasks[task].get_indices()

            # adjust the tasks
            # print(self.task_indices[task], len(self.task_indices[task]))
            # assert 1==0
            
            # only include good_cases
            # if start_index
            # filtered_cases = []
            # for i in self.task_indices[task]:
            #     if i in GOOD_CASES:
            #         filtered_cases.append(i)
            if cases:
                # self.remaining_tasks[agent][task] = sorted(list(set(CASES)))
                self.remaining_tasks[agent][task] = CASES
            else:
                if start_index is not None and end_index is not None:
                    self.remaining_tasks[agent][task] = self.task_indices[task].copy()[start_index:end_index]
                elif start_index is not None:
                    self.remaining_tasks[agent][task] = self.task_indices[task].copy()[start_index:]
                elif end_index is not None:
                    self.remaining_tasks[agent][task] = self.task_indices[task].copy()[:end_index]
                else:
                    self.remaining_tasks[agent][task] = self.task_indices[task].copy()
            
            # print(self.remaining_tasks[agent][task], type(self.remaining_tasks[agent][task]))
            # assert 1==0
            # self.remaining_tasks[agent][task] = self.task_indices[task].copy()[181:]
            if not os.path.exists(runs_file):
                continue
            with open(runs_file, "r") as f:
                for line in f:
                    try:
                        run = json.loads(line)
                        run.pop("time")
                        index = run.pop("index")
                        assert index is not None
                        run = TaskClientOutput.parse_obj(run)
                        assert isinstance(run.output, TaskOutput)
                    except:
                        continue
                    if index in self.remaining_tasks[agent][task]:
                        self.remaining_tasks[agent][task].remove(index)
                        self.record_completion(agent, task, index, run.output)
                    else:
                        print(
                            ColorMessage.yellow(
                                f"Warning: {agent}/{task}#{index} is finished, but not in the index list."
                            )
                        )
        print("self.tasks", self.tasks)
        count = sum(
            [
                len(self.remaining_tasks[agent][task])
                for agent in self.remaining_tasks
                for task in self.remaining_tasks[agent]
            ]
        )
        print(
            ColorMessage.cyan(f"Message: {count} samples remaining.")
        )

        for agent in self.remaining_tasks:
            agent_ = json.dumps(agent)
            tasks_ = len(self.remaining_tasks[agent])
            samples_ = sum(
                [
                    len(self.remaining_tasks[agent][task])
                    for task in self.remaining_tasks[agent]
                ]
            )
            if samples_ == 0:
                continue
            print(
                ColorMessage.cyan(
                    f"Agent {agent_} needs to run {tasks_} tasks with total {samples_} samples:"
                )
            )
            for task in self.remaining_tasks[agent]:
                print(
                    ColorMessage.cyan(
                        f"    Task {json.dumps(task)}: {len(self.remaining_tasks[agent][task])}"
                    )
                )

        # Create agents

        for agent in self.remaining_tasks:
            self.agents[agent] = self.config.definition.agent[agent].create()

    def get_output_dir(self, agent: str, task: str) -> str:
        return os.path.join(self.config.output, agent, task)

    def worker_generator(
        self, interval=10
    ) -> Iterator[Tuple[str, str, SampleIndex]]:

        node_list = ["SRC", "DST"]
        agent_node_index = {}
        task_node_index = {}
        for agent in self.agents:
            node_list.append(agent)
            agent_node_index[agent] = len(node_list) - 1
        for task in self.tasks:
            node_list.append(task)
            task_node_index[task] = len(node_list) - 1

        while True:

            # Step 0. Get real time task free worker
            print("Free workers per agent:", self.free_worker.agent)
            print("Free workers per task:", self.free_worker.task)
            with self.assignment_lock:
                for task in self.tasks:
                    self.free_worker.task[task] = self.tasks[task].get_concurrency()
                    print("tasks: ", task, self.free_worker.task[task]) 
                    # omnigibson 0
                    # assert 1==0
                print("Running Count: {}".format(self.running_count))

            # Step 1. init edges: SRC -> agent -> task -> DST
            print("Remaining tasks:", self.remaining_tasks)

            with self.assignment_lock:
                edges = {}
                for agent in self.agents:
                    edges[(0, agent_node_index[agent])] = self.free_worker.agent[agent]
                for task in self.tasks:
                    edges[(task_node_index[task], 1)] = self.free_worker.task[task]
                tot_remaining_samples = 0
                for agent in self.remaining_tasks:
                    for task in self.remaining_tasks[agent]:
                        tot_remaining_samples += len(self.remaining_tasks[agent][task])
                        edges[(agent_node_index[agent], task_node_index[task])] = len(
                            self.remaining_tasks[agent][task]
                        )
            if tot_remaining_samples == 0:
                if self.running_count == 0:
                    break
                else:
                    time.sleep(interval / 2 + random.random() * interval)
                    continue

            # Step 2. Create graph and calculate max flow

            print("node_list: ", node_list)
            print("edges: ", edges)

            graph = Graph(node_count=len(node_list), edges=edges)
            max_flow = MaxFlow(graph, src=0, dst=1)

            print("max_flow: ", max_flow.max_flow)

            if max_flow.max_flow == 0:
                time.sleep(interval / 2 + random.random() * interval)
                continue

            # Step 3. yield all (agent, task, index) tuples

            for (src, dst), e in max_flow.edges_dict.items():
                if (
                    src not in agent_node_index.values()
                    or dst not in task_node_index.values()
                ):
                    continue
                if e.flow == 0:
                    continue
                agent = node_list[src]
                task = node_list[dst]
                for _ in range(e.flow):
                    with self.assignment_lock:
                        index = self.remaining_tasks[agent][task].pop()
                        self.free_worker.agent[agent] -= 1
                        self.free_worker.task[task] -= 1
                    print(ColorMessage.green(f"Assigned {agent}/{task}#{index}"))
                    yield agent, task, index

            # Step 4. sleep for a while
            time.sleep(interval / 2 + random.random() * interval)

    def start(self, tqdm_out=None):
        self.started_count = sum(
            [
                len(self.remaining_tasks[agent][task])
                for agent in self.remaining_tasks
                for task in self.remaining_tasks[agent]
            ]
        )
        generator = self.worker_generator()
        self.overall_tqdm = tqdm(
            total=self.started_count,
            desc="Total",
            position=0,
            file=tqdm_out,
        )
        for idx, agent in enumerate(self.remaining_tasks.keys()):
            self.tqdm_ordered_by_agent[agent] = tqdm(
                total=sum(
                    [
                        len(self.remaining_tasks[agent][task])
                        for task in self.remaining_tasks[agent]
                    ]
                ),
                desc=agent,
                position=idx + 1,
                file=tqdm_out,
            )
        while True:
            try:
                agent, task, index = next(generator)
            except StopIteration:
                break
            self.start_worker(agent, task, index, self.finish_callback)

        self.overall_tqdm.close()
        for agent in self.tqdm_ordered_by_agent:
            self.tqdm_ordered_by_agent[agent].close()

        final_message = (
            "\n\n============================================\n"
            + ColorMessage.cyan(f"Message: {self.started_count} sample(s) started. ")
            + "\n"
            + ColorMessage.green(
                f"   >> {self.finished_count} sample(s) finished successfully."
            )
            + "\n"
        )
        if self.started_count != self.finished_count:
            final_message += (
                ColorMessage.red(
                    f"   >> {self.started_count - self.finished_count} sample(s) failed."
                )
                + "\n"
            )
        final_message += (
            ColorMessage.cyan(
                f"   >> results are saved to {self.config.output}"
            )
            + "\n"
        )
        final_message += "============================================\n\n"
        print(final_message)

    def record_completion(
        self, agent: str, task: str, index: SampleIndex, result: TaskOutput
    ):
        def calculate_overall_worker():
            nonlocal agent, task, index, result
            task_client = self.tasks[task]
            overall = task_client.calculate_overall(self.completions[agent][task])
            with open(
                os.path.join(self.get_output_dir(agent, task), "overall.json"), "w"
            ) as f:
                f.write(json.dumps(overall, indent=4, ensure_ascii=False))

        overall_calculation = False
        with self.assignment_lock:
            if agent not in self.completions:
                self.completions[agent] = {}
            if task not in self.completions[agent]:
                self.completions[agent][task] = []
            result.index = index
            self.completions[agent][task].append(result)
            if len(self.completions[agent][task]) == len(self.task_indices[task]):
                overall_calculation = True
        if overall_calculation:
            output_dir = self.get_output_dir(agent, task)
            if os.path.exists(os.path.join(output_dir, "overall.json")):
                return
            threading.Thread(target=calculate_overall_worker).start()

    def finish_callback(
        self, agent: str, task: str, index: SampleIndex, result: TaskClientOutput
    ):
        if result.error == TaskError.NOT_AVAILABLE.value:
            print(
                ColorMessage.yellow(
                    f"Warning: {task} is not available, retrying."
                )
            )
            with self.assignment_lock:
                self.remaining_tasks[agent][task].insert(0, index)
                self.free_worker.agent[agent] += 1
                self.free_worker.task[task] += 1
                self.running_count -= 1
            return

        if result.error is not None:
            print(ColorMessage.yellow(f"Warning: {agent}/{task}#{index} "
                                      f"failed with error {result.error} {result.info} {result.output}"))
            if self.auto_retry:
                with self.assignment_lock:
                    self.remaining_tasks[agent][task].insert(0, index)

        output_folder = self.get_output_dir(agent, task)
        os.makedirs(output_folder, exist_ok=True)
        timestamp: int = int(time.time() * 1000)
        time_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        write_to_file = (
            json.dumps(
                {
                    "index": index,
                    **result.dict(),
                    "time": {"timestamp": timestamp, "str": time_str},
                }
            )
            + "\n"
        )
        if not result.error:
            target_file = os.path.join(output_folder, "runs.jsonl")
            with self.assignment_lock:
                self.finished_count += 1
            self.record_completion(agent, task, index, result.output)
            self.overall_tqdm.update(1)
            self.tqdm_ordered_by_agent[agent].update(1)
        else:
            target_file = os.path.join(output_folder, "error.jsonl")
        with open(target_file, "a+", encoding="utf-8") as f:
            f.write(write_to_file)

        with self.assignment_lock:
            self.free_worker.agent[agent] += 1
            self.free_worker.task[task] += 1
            self.running_count -= 1

    def start_worker(
        self,
        agent: str,
        task: str,
        index: SampleIndex,
        finish_callback: Union[
            Callable[[str, str, SampleIndex, TaskClientOutput], None], None
        ] = None,
    ):
        
        def worker_thread():
            nonlocal agent, task, index, finish_callback

            result = self.tasks[task].run_sample(index, self.agents[agent])

            if finish_callback:
                finish_callback(agent, task, index, result)

        with self.assignment_lock:
            self.running_count += 1
        threading.Thread(target=worker_thread).start() 
        # Creates and starts a new thread that executes worker_thread, ensuring the task runs asynchronously.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="configs/assignments/default.yaml"
    )
    parser.add_argument(
        "--auto-retry", "-r", action="store_true", dest="retry"
    )
    parser.add_argument(
        "--cases", action="store_true"
    )
    parser.add_argument(
        "--start_index", "-s", type=int, default=None
    )
    parser.add_argument(
        "--end_index", "-e", type=int, default=None
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None
    )
    args = parser.parse_args()

    loader = ConfigLoader()
    config_ = loader.load_from(args.config)
    value = AssignmentConfig.parse_obj(config_)
    value = AssignmentConfig.post_validate(value)
    v = value.dict()
    

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    for assignment in value.assignments:
        agent = assignment.agent
        task = assignment.task
        if args.output_dir is not None:
            output_path = args.output_dir
        else:
            output_path = f"outputs/{timestamp}-{agent}"
        
        # Create directory
        os.makedirs(output_path, exist_ok=True)

        print(f"Assignment for {agent} ({task}) will output to: {output_path}")

        # Set global output (optional – depends if it's used elsewhere)
        value.output = output_path

        # Set per-task output_dir inside the task config
        if task in value.definition.task:
            value.definition.task[task].parameters['output_dir'] = output_path
        else:
            raise ValueError(f"Task '{task}' not found in value.definition.task")
    # print(value)
    # assert 1==0
    
    # import yaml

    # # Load the YAML
    # with open("configs/tasks/omnigibson.yaml", "r") as f:
    #     task_config = yaml.safe_load(f)

    # # Modify output_dir
    # task_config["omnigibson"]["parameters"]["output_dir"] = output_path

    # # Save it back
    # with open("configs/tasks/omnigibson.yaml", "w") as f:
    #     yaml.safe_dump(task_config, f)

    
    
    with std_out_err_redirect_tqdm() as orig_stdout:
        Assigner(value, args.retry, args.start_index, args.end_index, args.cases).start(tqdm_out=orig_stdout)
