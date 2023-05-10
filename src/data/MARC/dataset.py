import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torchtext
import torch.nn.utils.rnn as rnn_utils


class MARCDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data_path):
        self.data_path = data_path
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    data = json.loads(line)
                    break
        label = int(data["stars"])
        text = data["review_body"]
        return text, label

    def __len__(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            num_lines = sum([1 for line in f])
        print("num_lines ", num_lines)
        return int(num_lines)
