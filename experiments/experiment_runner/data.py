# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from experiments.experiment_runner.data_infrastructure import Data, Dataset


class DataSource:
    def get_data(self) -> Data:
        pass


class StaticDataSource(DataSource):

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def get_data(self) -> Dataset:
        return self._dataset

    @property
    def name(self):
        return self._dataset.name
