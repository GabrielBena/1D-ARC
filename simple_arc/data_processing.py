import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def load_single_file(file_path):

    with open(file_path, "r") as file:
        task = json.load(file)
        return task


def get_pandas_dataset(path):

    complete_dataset = []
    for n, task_name in enumerate(os.listdir(path)):
        for task_iter in os.listdir(f"{path}{task_name}/"):
            task = load_single_file(f"{path}/{task_name}/{task_iter}")
            for trial in ["train", "test"]:
                for t, single_task in enumerate(task[trial]):
                    complete_dataset.append(
                        {
                            "task_name": task_name,
                            "task_number": n,
                            "trial": trial,
                            "input": np.array(single_task["input"]),
                            "output": np.array(single_task["output"]),
                            "uuid": task["uuid"] if "uuid" in task.keys() else None,
                            "iteration": t,
                        }
                    )

    complete_dataset = pd.DataFrame(complete_dataset)
    max_shape, argmax_shape = (
        complete_dataset["input"].apply(lambda x: x.shape).max(),
        complete_dataset["input"].apply(lambda x: x.shape).argmax(),
    )

    for i in ["input", "output"]:

        complete_dataset[i] = complete_dataset[i].apply(
            lambda x: np.pad(
                x,
                [
                    (
                        (max_shape[0] - x.shape[0]) // 2,
                        (max_shape[0] - x.shape[0]) // 2,
                    ),
                    (
                        (max_shape[1] - x.shape[1]) // 2,
                        (max_shape[1] - x.shape[1]) // 2,
                    ),
                ],
                mode="constant",
                constant_values=(0, 0),
            )
        )

        # Finish in case there is some off by one error
        complete_dataset[i] = complete_dataset[i].apply(
            lambda x: np.pad(
                x,
                [[0, max_shape[0] - x.shape[0]], [0, max_shape[1] - x.shape[1]]],
                mode="constant",
                constant_values=(0, 0),
            )
        )

    return complete_dataset


def get_numpy_dataset(path):
    complete_dataset = get_pandas_dataset(path)

    return {
        "train": (
            {
                "input": np.stack(
                    complete_dataset.query("trial == 'train'")["input"].values
                ),
                "output": np.stack(
                    complete_dataset.query("trial == 'train'")["output"].values
                ),
                "tasks": np.array(
                    complete_dataset.query("trial == 'train'")["task_name"].values
                ),
                "task_number": np.array(
                    complete_dataset.query("trial == 'train'")["task_number"].values
                ),
            }
        ),
        "test": {
            "input": np.stack(
                complete_dataset.query("trial == 'test'")["input"].values
            ),
            "output": np.stack(
                complete_dataset.query("trial == 'test'")["output"].values
            ),
            "tasks": np.array(
                complete_dataset.query("trial == 'test'")["task_name"].values
            ),
            "task_number": np.array(
                complete_dataset.query("trial == 'test'")["task_number"].values
            ),
        },
    }
