from src.data.NLI.dataset import NLIDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

# Define the DataLoader class for NLI
class NLIDataLoader:
    def __init__(self, dataset, batch_size=32):
        # :param dataset: dataset to use
        # :param tokenizer: tokenizer to use
        # :param batch_size: batch size
        self.batch_size = batch_size
        print("Loading NLI dataset")
        self.dataset = NLIDataset(dataset)
        print("NLI dataset loaded")

    def get_dataloader(self):
        # :return: DataLoader for the dataset
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
