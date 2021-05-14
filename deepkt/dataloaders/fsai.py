import pandas as pd
import os
import pickle
from tqdm import tqdm
import numpy as np

from deepkt.dataloaders.base import BaseDataLoader


class FSAIDataLoader(BaseDataLoader):
    def __init__(self, config):
        """
        initialize the dataset, train_loader, test_loader
        :param config:
        """
        super().__init__(config)
        self.data_name = config["data_name"]
        data_path = f"../data/FSAI/{self.data_name}.pkl"
        data = pickle.load(open(data_path, "rb"))
        self.num_items = data["num_questions"]
        self.num_users = data["num_users"]
        print("num users: {}".format(self.num_users))
        print("num items: {}".format(self.num_items))

        self.generate_train_test_loaders(data)

    def finalize(self):
        pass
