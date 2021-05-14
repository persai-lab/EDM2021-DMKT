import numpy as np
from torch.utils.data import Dataset
from deepkt.datasets.transforms import SlidingWindow, Padding


class DKVMN_ExtDataset(Dataset):
    """
    prepare the data for data loader, including truncating long sequence and padding
    """

    def __init__(self, q_records, a_records, l_records, num_items, max_seq_len, min_seq_len=2,
                 max_subseq_len=2, stride=None, train=True, metric="auc"):
        """
        :param min_seq_len: used to filter out seq. less than min_seq_len
        :param max_seq_len: used to truncate seq. greater than max_seq_len
        :param max_subseq_len: used to truncate the lecture seq.
        """
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.num_items = num_items
        if stride is not None and train:
            self.stride = stride
        else:
            # because we pad 0 at the first element
            self.stride = max_seq_len - 1
        self.metric = metric

        self.q_data, self.a_data, self.l_data = self._transform(
            q_records, a_records, l_records, max_subseq_len)
        print("train samples.: {}".format(self.l_data.shape))
        self.length = len(self.q_data)

    def __len__(self):
        """
        :return: the number of training samples rather than training users
        """
        return self.length

    def __getitem__(self, idx):
        """
        for DKT, we apply one-hot encoding and return the idx'th data sample here, we dont
        need to return target, since the input of DKT also contain the target information
        reference: https://github.com/jennyzhang0215/DKVMN/blob/master/code/python3/model.py

        for SAKT, we dont apply one-hot encoding, instead we return the past interactions,
        current exercises, and current answers
        # reference: https://github.com/TianHongZXY/pytorch-SAKT/blob/master/dataset.py

        idx: sample index
        """
        questions = self.q_data[idx]
        answers = self.a_data[idx]
        lectures = self.l_data[idx]
        assert len(questions) == len(answers) == len(lectures)

        if self.metric == "rmse":
            interactions = []
            for i, q in enumerate(questions):
                interactions.append([q, answers[i]])
            interactions = np.array(interactions, dtype=float)
        else:
            interactions = np.zeros(self.max_seq_len, dtype=int)
            for i, q in enumerate(questions):
                interactions[i] = q + answers[i] * self.num_items
        target_answers = answers
        target_mask = (questions != 0)
        return interactions, lectures, questions, target_answers, target_mask

    def _transform(self, q_records, a_records, l_records=None, max_subseq_len=None):
        """
        transform the data into feasible input of model,
        truncate the seq. if it is too long and
        pad the seq. with 0s if it is too short
        """
        if l_records is not None and max_subseq_len is not None:
            assert len(q_records) == len(a_records) == len(l_records)
            l_data = []
            lec_sliding = SlidingWindow(self.max_seq_len, self.stride,
                                        fillvalue=[0] * max_subseq_len)
            padding = Padding(max_subseq_len, side='left', fillvalue=0)
        else:
            assert len(q_records) == len(a_records)

        q_data = []
        a_data = []
        # if seq length is less than max_seq_len, the sliding will pad it with fillvalue
        # the reason of inserting the first attempt with 0 and setting stride=self.max_seq_len-1
        # is to make sure every test point will be evaluated if a model cannot test the first
        # attempt
        sliding = SlidingWindow(self.max_seq_len, self.stride, fillvalue=0)
        for index in range(len(q_records)):
            q_list = q_records[index]
            a_list = a_records[index]
            q_list.insert(0, 0)
            a_list.insert(0, 0)
            assert len(q_list) == len(a_list)
            sample = {"q": q_list, "a": a_list}
            output = sliding(sample)
            q_data.extend(output["q"])
            a_data.extend(output["a"])
            if l_records is not None:
                l_list = l_records[index]
                l_list = [padding({"l": l[-max_subseq_len:]})["l"] for l in l_list]
                l_list.insert(0, [0] * max_subseq_len)
                assert len(q_list) == len(a_list) == len(l_list)
                sample = {"l": l_list}
                lec_output = lec_sliding(sample)
                l_data.extend(lec_output["l"])
                assert len(q_data) == len(a_data) == len(l_data)

        if l_records is not None:
            return np.array(q_data), np.array(a_data), np.array(l_data)
        else:
            return np.array(q_data), np.array(a_data)
