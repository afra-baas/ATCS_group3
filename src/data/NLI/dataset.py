import os
import json
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils

# Define the dataset class
class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset):
        # :param dataset: dataset to use
        # :param tokenizer: tokenizer to use
        self.dataset = dataset

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        # :param idx: index of the sample to fetch
        # :return: tuple (text, label) for the given index
        premise = self.dataset[idx]["premise"]
        hypothesis = self.dataset[idx]["hypothesis"]
        label = self.dataset[idx]["label"]
        return premise, hypothesis, label
