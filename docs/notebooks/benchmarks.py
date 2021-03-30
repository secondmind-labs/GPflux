# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Benchmarks

We benchmark GPflux' Deep GP on several UCI datasets.
The code to run the experiments can be found in `benchmarking/main.py`. The results are stored in `benchmarking/runs/*.json`. In this script we aggregate and plot the outcomes.
"""

# %% {"nbsphinx": "hidden"}
import glob
import json

import numpy as np
import pandas as pd

# %% {"nbsphinx": "hidden"}
LOGS = "../../benchmarking/runs/*.json"

data = []
for path in glob.glob(LOGS):
    with open(path) as json_file:
        data.append(json.load(json_file))

df = pd.DataFrame.from_records(data)
df = df.rename(columns={"model_name": "model"})

# %% {"nbsphinx": "hidden"}
table = df.groupby(["dataset", "model"]).agg(
    {
        "split": "count",
        **{metric: ["mean", "std"] for metric in ["mse", "nlpd"]},
    }
)
# %% [markdown]
"""
We report the mean and std. dev. of the MSE and Negative Log Predictive Density (NLPD) measured by running the experiment on 5 different splits. We use 90% of the data for training and the remaining 10% for testing. The output is normalised to have zero mean and unit variance.
"""
# %%
table
