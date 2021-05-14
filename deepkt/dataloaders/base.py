import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from deepkt.datasets.mlp_ext import MLP_ExtDataset
from deepkt.datasets.dkt import DKTDataset
from deepkt.datasets.dkt_ext import DKT_ExtDataset
from deepkt.datasets.dkvmn import DKVMNDataset
from deepkt.datasets.dkvmn_ext import DKVMN_ExtDataset
from deepkt.datasets.sakt import SAKTDataset
from deepkt.datasets.sakt_ext import SAKT_ExtDataset


class BaseDataLoader:
    """
    Base class for all data loaders
    """

    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.collate_fn = default_collate
        self.num_workers = config["num_workers"]

        self.validation_split = config["validation_split"]
        self.mode = config["mode"]
        self.agent = config["agent"]
        self.seed = config["seed"]
        self.metric = config["metric"]

        self.min_seq_len = config["min_seq_len"] if "min_seq_len" in config else None
        self.max_seq_len = config["max_seq_len"] if "max_seq_len" in config else None
        self.stride = config["stride"] if "stride" in config else None
        self.max_subseq_len = config["max_subseq_len"] if "max_subseq_len" in config else None

        self.num_items = None
        self.train_data = None
        self.train_loader = None
        self.test_data = None
        self.test_loader = None

        self.init_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }

    def generate_train_test_loaders(self, data):
        # agents that support multiple learning materials
        if self.agent in ["DMKTAgent", "MKT_IRTAgent", "DKVMN_ExtAgent", "DKVMN_IRT_ExtAgent",
                          "MLP_ExtAgent", "DKT_ExtAgent",
                          "SAKT_ExtAgent", "SAINT_ExtAgent", "AKT_ExtAgent"
                          ]:
            q_records = data["train"]["q_data"]
            a_records = data["train"]["a_data"]
            l_records = data["train"]["l_data"]
            if self.agent in ["DMKTAgent", "DKVMN_ExtAgent", "MKT_IRTAgent", "DKVMN_IRT_ExtAgent"]:
                self.train_data = DKVMN_ExtDataset(q_records, a_records, l_records,
                                                   self.num_items,
                                                   self.max_seq_len,
                                                   min_seq_len=self.min_seq_len,
                                                   max_subseq_len=self.max_subseq_len,
                                                   stride=self.stride,
                                                   train=True,
                                                   metric=self.metric)
            elif self.agent == "DKT_ExtAgent":
                self.train_data = DKT_ExtDataset(q_records, a_records, l_records, self.num_items,
                                                 self.max_seq_len,
                                                 min_seq_len=self.min_seq_len,
                                                 max_subseq_len=self.max_subseq_len,
                                                 stride=self.stride,
                                                 train=True,
                                                 metric=self.metric)

            elif self.agent in ["AKT_ExtAgent", "SAKT_ExtAgent", "SAINT_ExtAgent"]:
                self.train_data = SAKT_ExtDataset(q_records, a_records, l_records, self.num_items,
                                                  self.max_seq_len,
                                                  min_seq_len=self.min_seq_len,
                                                  max_subseq_len=self.max_subseq_len,
                                                  stride=self.stride,
                                                  train=True,
                                                  metric=self.metric)
            else:
                self.train_data = MLP_ExtDataset(q_records, a_records, l_records,
                                                 self.max_subseq_len)
            if self.mode == "train":
                self.init_kwargs["dataset"] = self.train_data
                n_samples = len(self.train_data)
                # split the train data into train and val sets based on the self.n_samples
                train_sampler, valid_sampler = self._split_sampler(n_samples, self.validation_split,
                                                                   self.seed)
                # turn off shuffle option which is mutually exclusive with sampler
                self.init_kwargs["shuffle"] = False
                # create the training loader
                self.train_loader = DataLoader(sampler=train_sampler, **self.init_kwargs)
                # create the validation that only do eval, so we set large batch size
                self.init_kwargs["batch_size"] = len(valid_sampler)
                self.test_loader = DataLoader(sampler=valid_sampler, **self.init_kwargs)
            elif self.mode in ["test", "predict"]:
                # if it is in test mode, we still need to train the model.
                # However, we combine train set and val set for training
                self.train_loader = DataLoader(self.train_data, **self.init_kwargs)

                q_records = data["test"]["q_data"]
                a_records = data["test"]["a_data"]
                l_records = data["test"]["l_data"]
                if self.agent in ["DMKTAgent", "DKVMN_ExtAgent", "MKT_IRTAgent",
                                  "DKVMN_IRT_ExtAgent"]:
                    self.test_data = DKVMN_ExtDataset(q_records, a_records, l_records,
                                                      self.num_items,
                                                      self.max_seq_len,
                                                      min_seq_len=self.min_seq_len,
                                                      max_subseq_len=self.max_subseq_len,
                                                      stride=self.stride,
                                                      train=False,
                                                      metric=self.metric)
                elif self.agent == "DKT_ExtAgent":
                    self.test_data = DKT_ExtDataset(q_records, a_records, l_records,
                                                    self.num_items,
                                                    self.max_seq_len,
                                                    min_seq_len=self.min_seq_len,
                                                    max_subseq_len=self.max_subseq_len,
                                                    stride=self.stride,
                                                    train=False,
                                                    metric=self.metric)

                elif self.agent in ["AKT_ExtAgent", "SAKT_ExtAgent", "SAINT_ExtAgent"]:
                    self.test_data = SAKT_ExtDataset(q_records, a_records, l_records,
                                                     self.num_items,
                                                     self.max_seq_len,
                                                     min_seq_len=self.min_seq_len,
                                                     max_subseq_len=self.max_subseq_len,
                                                     stride=self.stride,
                                                     train=False,
                                                     metric=self.metric)
                else:
                    self.test_data = MLP_ExtDataset(q_records, a_records, l_records,
                                                    self.max_subseq_len)
                self.init_kwargs["batch_size"] = len(self.test_data)
                self.test_loader = DataLoader(self.test_data, **self.init_kwargs)
            # elif self.mode == "predict":
            #     pass
            else:
                raise AttributeError

        # agents that support gradable learning materials
        elif self.agent in ["DKTAgent", "AKTAgent", "SAKTAgent", "SAINTAgent",
                            "DKVMNAgent", "DKVMN_IRTAgent", "DKVMN_IRT_PlusAgent",
                            "DKVMN_Distill_IRTAgent"]:
            q_records = data["train"]["q_data"]
            a_records = data["train"]["a_data"]
            if self.agent == "DKTAgent":
                self.train_data = DKTDataset(q_records, a_records, self.num_items,
                                             self.max_seq_len,
                                             min_seq_len=self.min_seq_len,
                                             stride=self.stride,
                                             train=True,
                                             metric=self.metric)
            elif self.agent in ["DKVMNAgent", "DKVMN_IRTAgent", "DKVMN_IRT_PlusAgent",
                                "DKVMN_Distill_IRTAgent"]:
                self.train_data = DKVMNDataset(q_records, a_records, self.num_items,
                                               self.max_seq_len,
                                               min_seq_len=self.min_seq_len,
                                               stride=self.stride,
                                               train=True,
                                               metric=self.metric)
            elif self.agent in ["AKTAgent", "SAKTAgent", "SAINTAgent"]:
                self.train_data = SAKTDataset(q_records, a_records, self.num_items,
                                              self.max_seq_len,
                                              min_seq_len=self.min_seq_len,
                                              stride=self.stride,
                                              train=True,
                                              metric=self.metric)
            if self.mode == "train":
                self.init_kwargs["dataset"] = self.train_data
                n_samples = len(self.train_data)
                # split the train data into train and val sets based on the self.n_samples
                train_sampler, valid_sampler = self._split_sampler(n_samples, self.validation_split,
                                                                   self.seed)
                # turn off shuffle option which is mutually exclusive with sampler
                self.init_kwargs["shuffle"] = False
                # create the training loader
                self.train_loader = DataLoader(sampler=train_sampler, **self.init_kwargs)
                # create the validation that only do eval, so we set large batch size
                self.init_kwargs["batch_size"] = len(valid_sampler)
                self.test_loader = DataLoader(sampler=valid_sampler, **self.init_kwargs)
            elif self.mode in ["test", "predict"]:
                # if it is in test mode, we still need to train the model.
                # However, we combine train set and val set for training
                self.train_loader = DataLoader(self.train_data, **self.init_kwargs)

                q_records = data["test"]["q_data"]
                a_records = data["test"]["a_data"]
                if self.agent == "DKTAgent":
                    self.test_data = DKTDataset(q_records, a_records, self.num_items,
                                                self.max_seq_len,
                                                min_seq_len=self.min_seq_len,
                                                stride=self.stride,
                                                train=False,
                                                metric=self.metric)
                elif self.agent in ["DKVMNAgent", "DKVMN_IRTAgent", "DKVMN_IRT_PlusAgent",
                                    "DKVMN_Distill_IRTAgent"]:
                    self.test_data = DKVMNDataset(q_records, a_records, self.num_items,
                                                  self.max_seq_len,
                                                  min_seq_len=self.min_seq_len,
                                                  stride=self.stride,
                                                  train=False,
                                                  metric=self.metric)
                elif self.agent in ["AKTAgent", "SAKTAgent", "SAINTAgent"]:
                    self.test_data = SAKTDataset(q_records, a_records, self.num_items,
                                                 self.max_seq_len,
                                                 min_seq_len=self.min_seq_len,
                                                 stride=self.stride,
                                                 train=False,
                                                 metric=self.metric)
                self.init_kwargs["batch_size"] = len(self.test_data)
                self.test_loader = DataLoader(self.test_data, **self.init_kwargs)
            # elif self.mode == "predict":
            #     pass
            else:
                raise AttributeError

    def _split_sampler(self, n_samples, split, seed):
        if split == 0.0:
            return None, None

        idx_full = np.arange(n_samples)

        np.random.seed(seed)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < n_samples, \
                "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def finalize(self):
        pass

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
