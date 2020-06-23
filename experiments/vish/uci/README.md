# A hacker's way to running experiments [WIP]

by Vincent@prowler.io

** !! DISCLAIMER: Very Quick and Dirty Draft !! **

Over the last couple of years I've been running quite a few experiments. Some of my setups have worked better than others. In this doc I describe a process that I've been iterating over for a while, and is working well for me. There is nothing revolutionary about the process, but it should be very easy for you to get going, and adopt to your needs!

There are basically three parts to this:
1. Where do you get your data from (and who pre-processes it)?
2. An experiment: Specify a model, train and evaluate it.
   1. [Optional] Run multiple experiments in parallel.
3. Analyzing results

## Data

[Bayesian benchmarks](https://github.com/hughsalimbeni/bayesian_benchmarks) is a simple way of giving your script access to the UCI datasets like Power, Kin8mn, Energy, Yacht, etc. Bayesian Benchmarks downloads the raw data (if a URL is specified), normalizes it and splits it in a train and test split. Bayesian Benchmarks has more features, but I don't use them. The repo can be clone with
```
git clone git@github.com:hughsalimbeni/bayesian_benchmarks.git
```
Quick example on how to use it:
```python
from bayesian_benchmarks.data import Energy
data = Energy(split=0, prop=0.9)  # First fold with 90% training and 10% test.
data_train = (data.X_train, data.Y_train)
data_test = (data.X_test, data.Y_test)
data.N, data.D  # total number of datapoints, dimensionality of the input data
```

### Wilson's datasets
If you're interested in running Wilson's UCI datasets (e.g. KEGGU, Protein, Buzz, Song, Stock, HouseElectric, etc.) you have to [download them manually](https://drive.google.com/open?id=0BxWe_IuTnMFcYXhxdUNwRHBKTlU). Unzip the file `tar -xvzf uci.tar.gz` and move the content to inside Bayesian benchmarks `mv uci bayesian_benchmarks/bayesian_benchmarks/data/`.

Note:
- some papers (e.g. Exact GP on Million of datapoints [Wang, 2019] and SOLVE-GP [Shu, 2020]) report HouseElectric to have 9 dimensions, according to my investigation this is not true - it has 11.
- some papers report the number of datapoints `N` in their table as the training set size (`total_number_of_datapoints * train_fraction`), while other use just `N = total_number_of_datapoints`. 


## Experiment

An experiment consists of a single run of a model on one dataset and one split.

The code that loads the data, build, fits and evaluates the model lives in [`./main.py`](./main.py).
The script uses [sacred](https://github.com/IDSIA/sacred) to configure and monitor the experiment, which can be installed from pypi:
```
pip install sacred
```

The script can be executed using the command line:
```
python main -p with option1=value1 option2=value2 ...
```
where `option1` can be, for example, the kernel type (`kernel_type`), `dataset`, etc. the model.

The output of the script is written to `LOGS` specified in [`./main.py`](./main.py). The script stores a dictionary in JSON format with the key metrics and configuration settings. This file can be processed later when aggregating the results.

### Executing multiple experiments in parallel

The code in [`create_experiments.py`](./create_experiments.py) can be used to create a `commands.txt` file listing all the different experiments that need running for a single table/experiment (e.g. different datasets and splits.).
Example of `commands.txt`
```
python main.py -p with  dataset=Wilson_protein split=0;
python main.py -p with  dataset=Wilson_protein split=1;
python main.py -p with  dataset=Wilson_slice split=0;
python main.py -p with  dataset=Wilson_slice split=1;
```

We use a [simple scheduler](https://pypi.org/project/simple-gpu-scheduler/) to run our commands on separate GPUs in parallel.
The package can be installed from pypi
```
pip install simple_gpu_scheduler
```

We can feed to output file `commands.txt` from [`create_experiments.py`](./create_experiments.py) to the scheduler
```
simple_gpu_scheduler --gpus 0,1,2 < commands.txt
```
specifying the use of GPU 0, 1 and 2.


## Visualising the results

The output from each experiment (produced by [`./main.py`](./main.py)) are stored as JSON files in the `LOGS` directory (`LOGS` is specified in `main.py`) and can best be viewed using a simple streamlit app. The code for this app lives in [`view_results.py`](./view_results.py). It simply allows you to specify a regex to the path containing the result jsons, which it uses to collect and display all the results.

Install streamlit from pypi
```
pip install streamlit
```
and run the app 
```
streamlit run view_results.py
```
The app will give you a URL you can access through your browser.



## Other tools

There are more tools I think can be quite handy at times and maybe less known:

1. `tmux`: (successor of `screens`)
2. `ngrok`