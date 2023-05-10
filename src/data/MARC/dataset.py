import os
import json
import torch

from datasets import load_dataset


class MARCDataset(torch.utils.data.Dataset):
    def __init__(self, language="fr"):
        # :param dataset: dataset to use
        # :param tokenizer: tokenizer to use
        self.language = language
        self.dataset = load_dataset("amazon_reviews_multi", language).with_format(
            "torch"
        )

    def __getitem__(self, idx: int):
        label = self.dataset['starts']
        text = self.dataset['text']
        return text, label

    def __len__(self):
        return len(self.dataset)
