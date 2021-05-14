"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import logging
import torch
import shutil
import numpy as np
from sklearn import metrics
import pickle
import torch.nn as nn

from tensorboardX.writer import SummaryWriter
from deepkt.utils.metrics import AverageMeter, AverageMeterList


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        self.current_epoch = None
        self.current_iteration = None
        self.model = None
        self.optimizer = None
        self.data_loader = None

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = config.seed
        self.mode = config.mode
        self.device = torch.device("cpu")

        # Summary Writer
        self.summary_writer = None
        self.true_labels = None
        self.pred_labels = None
        self.best_epoch = None
        self.train_loss = None
        self.train_loss_list = []
        self.best_train_loss = None
        self.best_val_perf = None
        self.metric = config.metric
        self.save = config.save_checkpoint
        if self.metric == "rmse":
            self.best_val_perf = 1.
        elif self.metric == "auc":
            self.best_val_perf = 0.
        else:
            raise AttributeError
        if "target_train_loss" in config:
            self.target_train_loss = config.target_train_loss
        else:
            self.target_train_loss = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info(f"Checkpoint loaded successfully from '{self.config.checkpoint_dir}' "
                             f"at (epoch {checkpoint['epoch']}) at (iteration "
                             f"{checkpoint['iteration']})\n")
        except OSError as e:
            self.logger.info(f"No checkpoint exists from '{self.config.checkpoint_dir}'. "
                             f"Skipping...")
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is
            the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def save_results(self):
        torch.save(self.true_labels, self.config.out_dir + "true_labels.tar")
        torch.save(self.pred_labels, self.config.out_dir + "pred_labels.tar")

    def track_best(self, true_labels, pred_labels):
        self.pred_labels = np.array(pred_labels).squeeze()
        self.true_labels = np.array(true_labels).squeeze()
        self.logger.info(
            "pred size: {} true size {}".format(self.pred_labels.shape, self.true_labels.shape))
        if self.metric == "rmse":
            perf = np.sqrt(metrics.mean_squared_error(self.true_labels, self.pred_labels))
            self.logger.info('RMSE: {:.05}'.format(perf))
            if perf < self.best_val_perf:
                self.best_val_perf = perf
                self.best_train_loss = self.train_loss.item()
                self.best_epoch = self.current_epoch
        elif self.metric == "auc":
            perf = metrics.roc_auc_score(self.true_labels, self.pred_labels)
            prec, rec, _ = metrics.precision_recall_curve(self.true_labels, self.pred_labels)
            pr_auc = metrics.auc(rec, prec)
            self.logger.info('ROC-AUC: {:.05}'.format(perf))
            self.logger.info('PR-AUC: {:.05}'.format(pr_auc))
            if perf > self.best_val_perf:
                self.best_val_perf = perf
                self.best_train_loss = self.train_loss.item()
                self.best_epoch = self.current_epoch
        else:
            raise AttributeError

    def early_stopping(self):
        if self.mode == "test":
            if self.target_train_loss is not None and self.train_loss <= self.target_train_loss:
                # early stop, target train loss comes from hyperparameters tuning step.
                self.logger.info("Early stopping...")
                self.logger.info("Target Train Loss: {}".format(self.target_train_loss))
                self.logger.info("Current Train Loss: {}".format(self.train_loss))
                return True
            # elif self.current_epoch > 10:
            #     if self.train_loss > torch.mean(self.train_loss_list[-10:]):
            #         return True
            # else:
            #     self.train_loss_list.append(self.train_loss)

    def run(self):
        """
        The main operator
        :return:
        """
        if self.mode in ["train", "test"]:
            try:
                self.train()
            except KeyboardInterrupt:
                self.logger.info("You have entered CTRL+C.. Wait to finalize")
        elif self.mode == "predict":
            self.predict()
        else:
            print(self.mode)
            raise ValueError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process.py the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.logger.info("Saving checkpoint...")
        if self.save is True:
            self.save_checkpoint()
            self.save_results()
        # self.summary_writer.export_scalars_to_json(
        #     "{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
        self.data_loader.finalize()
        return self.best_epoch, self.best_train_loss, self.best_val_perf



    # def depict_knowledge_quiz_only(self, q_data, qa_data, l_data, idx):
    #     if self.metric == "rmse":
    #         qa_data = qa_data.float()
    #     # batch_size, seq_len = q_data.size(0), q_data.size(1)
    #     batch_size, seq_len, lec_len = l_data.size(0), l_data.size(1), l_data.size(2)
    #     self.model.value_matrix = torch.Tensor(self.model.num_concepts, self.model.value_dim).to(
    #         self.device)
    #     nn.init.normal_(self.model.value_matrix, mean=0., std=self.model.init_std)
    #     self.model.value_matrix = self.model.value_matrix.clone().repeat(batch_size, 1, 1)
    #
    #     q_embed_data = self.model.q_embed_matrix(q_data)
    #     qa_embed_data = self.model.qa_embed_matrix(qa_data)
    #     # split the data seq into chunk and process.py each question sequentially
    #     sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
    #     sliced_qa_embed_data = torch.chunk(qa_embed_data, seq_len, dim=1)
    #
    #     l_read_content = torch.Tensor(batch_size, self.model.value_dim).to(self.device)
    #     ls = torch.Tensor(batch_size, self.model.value_dim).to(self.device)
    #     sliced_l_data = torch.chunk(l_data, seq_len, dim=1)  # seq_len * (batch_size, 1, lec_len)
    #
    #     knowledge_state = []
    #     for i in range(seq_len):
    #         qid = q_data.squeeze()[i]
    #         print("question: {}".format(qid))
    #         if qid == 0:
    #             continue
    #
    #         q = sliced_q_embed_data[i].squeeze(1)  # (batch_size, key_dim)
    #         qa = sliced_qa_embed_data[i].squeeze(1)  # (batch_size, key_dim)
    #         q_correlation_weight = self.model.compute_correlation_weight(q)
    #         q_read_content = self.model.read(q_correlation_weight)
    #
    #         masked_summary_fc = nn.Linear(2 * self.model.key_dim + 2 * self.model.value_dim,
    #                                       self.model.summary_dim).to(self.device)
    #         mask_used = torch.zeros(
    #             self.model.summary_dim,
    #             2 * self.model.key_dim + 2 * self.model.value_dim
    #         ).to(self.device)
    #         mask_used[:, :self.model.key_dim] = 1.
    #
    #         # print(self.model.state_dict()["summary_fc.weight"].shape)
    #         # print(mask_used.shape)
    #         masked_summary_fc.weight.data = self.model.state_dict()["summary_fc.weight"] * mask_used
    #         masked_summary_fc.bias.data = self.model.state_dict()["summary_fc.bias"]
    #         cws = torch.eye(self.model.num_concepts).to(self.device)
    #         current_state = []
    #         for cw in cws:
    #             nn.init.zeros_(l_read_content)
    #             nn.init.zeros_(ls)
    #             read_content = self.model.read(cw)
    #             mastery_level = torch.cat([read_content, q, l_read_content, ls], dim=1)
    #             summary_output = self.model.tanh(masked_summary_fc(mastery_level))
    #             batch_sliced_pred = self.model.sigmoid(self.model.linear_out(summary_output))
    #             current_state.append(batch_sliced_pred.squeeze().item())
    #         knowledge_state.append(current_state)
    #         self.model.value_matrix = self.model.write(q_correlation_weight, qa)
    #
    #         # after update value_matrix with qa_data, we test the knowledge
    #         # should not change the order of code
    #         nn.init.zeros_(l_read_content)
    #         nn.init.zeros_(ls)
    #         # (batch_size, 128, value_dim)
    #         l_embed_data = self.model.l_embed_matrix(sliced_l_data[i].squeeze(1).long())
    #         # print(l_embed_data.shape)
    #         sliced_l_embed_data = torch.chunk(l_embed_data, lec_len, dim=1)
    #         # 128 * (batch_size, 1, value_dim)
    #
    #
    #         # masked_summary_fc = nn.Linear(2 * self.model.key_dim + 2 * self.model.value_dim,
    #         #                               self.model.summary_dim).to(self.device)
    #         # mask_used = torch.zeros(self.model.summary_dim,
    #         #                         2 * self.model.key_dim + 2 * self.model.value_dim).to(
    #         #     self.device)
    #         # mask_used[:, 2 * self.model.key_dim:3 * self.model.key_dim] = 1.
    #         # masked_summary_fc.weight.data = self.model.state_dict()["summary_fc.weight"] * mask_used
    #         # masked_summary_fc.bias.data = self.model.state_dict()["summary_fc.bias"]
    #         #
    #         # cws = torch.eye(self.model.num_concepts).to(self.device)
    #         # current_state = []
    #         # for cw in cws:
    #         #     read_content = self.model.read(cw)
    #         #     mastery_level = torch.cat([q_read_content, q, read_content, ls], dim=1)
    #         #     summary_output = self.model.tanh(masked_summary_fc(mastery_level))
    #         #     batch_sliced_pred = self.model.sigmoid(self.model.linear_out(summary_output))
    #         #     current_state.append(batch_sliced_pred.squeeze().item())
    #         # knowledge_state.append(current_state)
    #
    #         nn.init.zeros_(l_read_content)
    #         nn.init.zeros_(ls)
    #         # (batch_size, 128, value_dim)
    #         l_embed_data = self.model.l_embed_matrix(sliced_l_data[i].squeeze(1).long())
    #         # print(l_embed_data.shape)
    #         sliced_l_embed_data = torch.chunk(l_embed_data, lec_len, dim=1)
    #         # 128 * (batch_size, 1, value_dim)
    #         for j in range(lec_len):
    #             l = sliced_l_embed_data[j].squeeze(1)  # (batch_size, value_dim)
    #             l_correlation_weight = self.model.compute_correlation_weight(l)
    #             l_read_content += self.model.read(l_correlation_weight)
    #             self.model.value_matrix = self.model.write(l_correlation_weight, l)
    #             ls += l
    #
    #         # get the knowledge transition over different lectures
    #         # print("lectures: {}".format(sliced_l_data[i].squeeze(1)))
    #         # for j in range(lec_len):
    #         #     if sliced_l_data[i].squeeze(1).squeeze(0)[j] == 0:
    #         #         continue
    #         #
    #         #     current_state = []
    #         #     cws = torch.eye(self.model.num_concepts).to(self.device)
    #         #     for cw in cws:
    #         #         read_content = self.model.read(cw)
    #         #         mastery_level = torch.cat([q_read_content, q, read_content, ls], dim=1)
    #         #         summary_output = self.model.tanh(masked_summary_fc(mastery_level))
    #         #         batch_sliced_pred = self.model.sigmoid(self.model.linear_out(summary_output))
    #         #         current_state.append(batch_sliced_pred.squeeze().item())
    #         #
    #         #     l = sliced_l_embed_data[j].squeeze(1)  # (batch_size, value_dim)
    #         #     l_correlation_weight = self.model.compute_correlation_weight(l)
    #         #     l_read_content = self.model.read(l_correlation_weight)
    #         #     self.model.value_matrix = self.model.write(l_correlation_weight, l)
    #         #     ls += l
    #         #     knowledge_state.append(current_state)
    #
    #     knowledge_state = np.array(knowledge_state)
    #     print(knowledge_state.shape)
    #     torch.save(knowledge_state, self.config.out_dir + "K_{}.pkl".format(idx))
