import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torchtext
import torch.nn.utils.rnn as rnn_utils
import random
from datasets import load_dataset


# Define the dataset class
class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, language='en'):
        self.dataset = load_dataset("xnli", language).with_format(
            "torch"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        premise = self.dataset[idx]['premise']
        hypothese = self.dataset[idx]['hypothesis']
        label = self.dataset[idx]['label']
        return (premise, hypothese), label


# Define the DataLoader
def create_dataloader_nli(dataset, batch_size=32):
    dataset = NLIDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True,  num_workers=4)
    return dataloader


# class MARCDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path, tokenizer):
#         self.data_path = data_path
#         self.tokenizer = tokenizer

#     def __getitem__(self, idx):
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             for i, line in enumerate(f):
#                 if i == idx:
#                     data = json.loads(line)
#                     break
#         label = int(data['stars'])
#         text = data['review_body']
#         return text, label

#     def __len__(self):
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             num_lines = sum([1 for line in f])
#         print('num_lines ', num_lines)
#         return int(num_lines)

# def get_random_sample(self, sample_size=100, seed=42):
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             data_lines = f.readlines()
#         random.seed(seed)
#         sample_lines = random.sample(data_lines, sample_size)
#         sample_dataset = [json.loads(line) for line in sample_lines]
#         sample_labels = [int(data['stars']) for data in sample_dataset]
#         sample_texts = [data['review_body'] for data in sample_dataset]
#         return sample_texts, sample_labels

class MARCDataset(torch.utils.data.Dataset):
    def __init__(self, language="en"):
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
    
    def get_random_sample(self, sample_size=100, seed=42):
        random.seed(seed)
        sample_indices = random.sample(range(len(self.dataset)), sample_size)
        sample_texts = [self.dataset[i]["review_body"] for i in sample_indices]
        sample_labels = [self.dataset[i]["stars"] for i in sample_indices]
        return sample_texts, sample_labels



def create_dataloader(batch_size=32):
    marc_dataset = MARCDataset()

    # Get a random sample of 1000 items
    sample_texts, sample_labels = marc_dataset.get_random_sample(
        sample_size=100, seed=42)

    # Create a new dataset and dataloader from the sample
    sample_dataset = [(text, label)
                      for text, label in zip(sample_texts, sample_labels)]
    marc_dataloader = torch.utils.data.DataLoader(
        sample_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)

    # marc_dataloader = torch.utils.data.DataLoader(
    #     marc_dataset, batch_size=batch_size, shuffle=True)
    return marc_dataloader
