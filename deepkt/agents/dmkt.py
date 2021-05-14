import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
import sys

from deepkt.agents.base import BaseAgent
from deepkt.graphs.models.dmkt import DMKT

# should not remove import statements below, it;s being used seemingly.
from deepkt.dataloaders import *

cudnn.benchmark = True
from deepkt.utils.misc import print_cuda_statistics
import warnings

warnings.filterwarnings("ignore")


# Notes on training DKVMN:
#   1. the batch size should not be large
#   2. the learning rate should not be large
#   3. should clip the gradient to avoid gradient vanishing or exploding
#   4. setup a good parameter initializer for key and value memory
#   5. use learning rate scheduler or reduce learning rate adaptively


class DMKTAgent(BaseAgent):
    def __init__(self, config):
        """initialize the agent with provided config dict which inherent from the base agent
        class"""
        super().__init__(config)

        # initialize the data_loader, which include preprocessing the data
        data_loader = globals()[config.data_loader]  # remember to import the dataloader
        self.data_loader = data_loader(config=config)
        # self.data_loader have attributes: train_data, train_loader, test_data, test_loader
        # note that self.data_loader.train_data is same as self.data_loader.train_loader.dataset
        self.mode = config.mode
        self.metric = config.metric

        config.num_items = self.data_loader.num_items
        config.num_nongradable_items = self.data_loader.num_nongradable_items
        self.model = DMKT(config)
        self.criterion = nn.BCELoss(reduction='sum')
        if config.optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.config.learning_rate,
                                       momentum=self.config.momentum)
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.config.learning_rate,
                                        eps=self.config.epsilon)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )
        # self.scheduler = optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=config.max_learning_rate,
        #     steps_per_epoch=len(self.data_loader.train_loader),
        #     epochs=config.max_epoch
        # )

        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer,
        #     milestones=[2000, 4000, 6000, 8000, 10000],
        #     gamma=0.667
        # )

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        # this loading should be after checking cuda
        self.load_checkpoint(self.config.checkpoint_file)

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1
            if self.early_stopping():
                break

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        self.logger.info("\n")
        self.logger.info("Train Epoch: {}".format(self.current_epoch))
        self.logger.info("learning rate: {}".format(self.optimizer.param_groups[0]['lr']))
        self.train_loss = 0
        train_elements = 0
        for batch_idx, data in enumerate(tqdm(self.data_loader.train_loader)):
            interactions, lec_interactions_list, questions, target_answers, target_mask = data
            interactions = interactions.to(self.device)
            lec_interactions_list = lec_interactions_list.to(self.device)
            questions = questions.to(self.device)
            target_answers = target_answers.to(self.device)
            target_mask = target_mask.to(self.device)
            self.optimizer.zero_grad()  # clear previous gradient
            # need to double check the target mask
            output = self.model(questions, interactions, lec_interactions_list)
            # print("target answer {}".format(target_answers))
            label = torch.masked_select(target_answers, target_mask)
            # print("output: {}".format(output))
            output = torch.masked_select(output, target_mask)
            loss = self.criterion(output.float(), label.float())
            # should use reduction="mean" not "sum", otherwise, performance drops significantly
            self.train_loss += loss.item()
            train_elements += target_mask.int().sum()
            loss.backward()  # compute the gradient

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()  # update the weight
            # self.scheduler.step()  # for CycleLR Scheduler or MultiStepLR
            self.current_iteration += 1
        # used for ReduceLROnPlateau
        self.train_loss = self.train_loss / train_elements
        self.scheduler.step(self.train_loss)
        self.logger.info("Train Loss: {:.6f}".format(self.train_loss))

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        if self.mode == "train":
            self.logger.info("Validation Result at Epoch: {}".format(self.current_epoch))
        else:
            self.logger.info("Test Result at Epoch: {}".format(self.current_epoch))
        test_loss = 0
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for data in self.data_loader.test_loader:
                interactions, lec_interactions_list, questions, target_answers, target_mask = data
                interactions = interactions.to(self.device)
                lec_interactions_list = lec_interactions_list.to(self.device)
                questions = questions.to(self.device)
                target_answers = target_answers.to(self.device)
                target_mask = target_mask.to(self.device)
                output = self.model(questions, interactions, lec_interactions_list)
                output = torch.masked_select(output[:, 1:], target_mask[:, 1:])
                label = torch.masked_select(target_answers[:, 1:], target_mask[:, 1:])
                # output = torch.masked_select(output, target_mask)
                # label = torch.masked_select(target_answers, target_mask)
                test_loss += self.criterion(output.float(), label.float()).item()
                pred_labels.extend(output.tolist())
                true_labels.extend(label.tolist())
                # print(list(zip(true_labels, pred_labels)))
        self.track_best(true_labels, pred_labels)

