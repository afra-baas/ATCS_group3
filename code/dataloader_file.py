import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# Define the dataset
class SSTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.sentences = dataset['sentence']
        self.labels = dataset['label']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode_plus(
            self.sentences[idx],
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }


# Define the DataLoader
def get_dataloader(dataset, tokenizer, batch_size=32):
    dataset = SSTDataset(dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
