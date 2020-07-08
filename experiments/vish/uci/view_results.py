import glob
import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from bayesian_benchmarks import data as uci_datasets


def get_dataset_class(dataset):
    return getattr(uci_datasets, dataset)


st.title("Visualise results")

file_regex = st.text_input(label="Specify location of JSON files", value="./logs/*/*",)
print(file_regex)

data = []
paths = []
for path in glob.glob(file_regex + ".json"):
    paths.append(path)
    with open(path) as json_file:
        data.append(json.load(json_file))

if len(paths) == 0:
    st.write("No results found - try a different regex")

else:
    st.write("Results found:", len(data))
    import pprint

    pprint.pprint(paths)

    # Show all ############################
    df = pd.DataFrame.from_records(data)
    df["paths"] = paths
    df["N"] = [get_dataset_class(d).N for d in df.dataset]
    df["D"] = [get_dataset_class(d).D for d in df.dataset]
    st.write("Raw results")
    st.write(df)

    # Aggregate ############################
    st.write("Aggregate")

    default_groupby_keys = ["model", "dataset"]
    average_over = "split"
    all_metrics = ["mse", "rmse", "nlpd", "time"]
    all_datasets = list(df.dataset.unique())
    all_models = list(df.model.unique())
    print(all_datasets)

    groupby_keys = st.multiselect(
        label="Group by", options=list(df.columns), default=default_groupby_keys,
    )

    selected_metrics = st.multiselect(
        label="Metrics", options=all_metrics, default=all_metrics,
    )
    selected_datasets = st.multiselect(
        label="Datasets", options=all_datasets, default=all_datasets,
    )

    selected_models = st.multiselect(
        label="Models", options=all_models, default=all_models,
    )

    all_unique_elements = ["N", "D", "M"]

    df = (
        df[df.dataset.isin(selected_datasets) & df.model.isin(selected_models)]
        .groupby(groupby_keys)
        .agg(
            {
                average_over: "count",
                **{element: "max" for element in all_unique_elements},
                **{metric: ["mean", "std"] for metric in selected_metrics},
            }
        )
    )

    st.write(df)
