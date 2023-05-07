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
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        premise = self.dataset[idx]['premise']
        hypothese = self.dataset[idx]['hypothesis']
        label = self.dataset[idx]['label']
        # inputs = self.tokenizer(
        #     text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return premise, hypothese, label


# Define the DataLoader
def create_dataloader_nli(dataset, tokenizer, lang, batch_size=32):
    dataset = NLIDataset(dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class MARCDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == idx:
                    data = json.loads(line)
                    break
        label = int(data['stars'])
        text = data['review_body']
        # text = self.tokenizer(text)
        # print(text)
        return text, label

    def __len__(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for line in f)
        return num_lines


def create_dataloader(batch_size, language, dataset_type, tokenizer):

    current_dir = os.getcwd()
    languages = {"English": 'en', "German": 'de', "Spanish": 'es',
                 "French": 'fr', "Japanese": "ja", "Chinese": "zh"}
    data_path = current_dir + '/ATCS_group3/marc_data/dataset_{}_{}.json'.format(
        languages[language], dataset_type)
    print(data_path)
    # tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    marc_dataset = MARCDataset(data_path, tokenizer)
    # print(marc_dataset[0])
    marc_dataloader = torch.utils.data.DataLoader(
        marc_dataset, batch_size=batch_size, shuffle=True)
    return marc_dataloader
