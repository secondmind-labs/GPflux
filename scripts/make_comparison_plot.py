import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

AUTOML_GLOBAL_DATA_ROOT = os.environ["AUTOML_GLOBAL_DATA_ROOT"]

tag = "experiment"
datasets = ["boston"]
models = ["linear", "SVGPexample", "DGP2017"]
num_folds = 10

model_name_conversion = {"linear": "a) Linear", "SVGPexample": "b) SVGP", "DGP2017": "c) DGP"}

for dataset in datasets:
    dataset_results = {}
    for model in models:
        fold_results = []
        for fold_index in range(num_folds):
            experiment_dir_glob = f"{tag}-{model}-{dataset}_{num_folds}.{fold_index}-*"
            full_report_glob = os.path.join(
                "model_training_tasks", "*", "tables", "full_report.csv"
            )
            glob_pattern = os.path.join(
                AUTOML_GLOBAL_DATA_ROOT, "experiments", experiment_dir_glob, full_report_glob,
            )
            glob_result = glob.glob(glob_pattern)

            if not glob_result:
                print(f"{dataset} / {model}: missing fold index {fold_index} / {num_folds}")
                continue
            if len(glob_result) > 1:
                raise ValueError(
                    f"{dataset} / {model}: found more than one result for fold index {fold_index} / {num_folds} - please remove erroneous ones!"
                )

            [filename] = glob_result
            res = pd.read_csv(filename)
            fold_results.append(res)

        model_df = pd.concat(fold_results)
        model_df["model"] = model_name_conversion[model]
        dataset_results[model] = model_df

    dataset_df = pd.concat([dataset_results[model] for model in models])

    for i, metric in enumerate(["rmse", "nlpd"]):
        columns = [f"train_{metric}", f"test_{metric}"]
        dataset_df.boxplot(by="model", column=columns, grid=False)

plt.show()
