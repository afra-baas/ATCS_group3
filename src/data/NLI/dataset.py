import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torchtext
import torch.nn.utils.rnn as rnn_utils

# Define the dataset class
class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        # :param dataset: dataset to use
        # :param tokenizer: tokenizer to use
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple(str, str, int):
        # :param idx: index of the sample to fetch
        # :return: tuple (text, label) for the given index
        premise = self.dataset[idx]["premise"]
        hypothese = self.dataset[idx]["hypothesis"]
        label = self.dataset[idx]["label"]
        # inputs = self.tokenizer(
        #     text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return premise, hypothese, label
