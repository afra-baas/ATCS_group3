from src.data.NLI.dataset import NLIDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

# Define the DataLoader class for NLI
class NLIDataLoader:
    def __init__(self, language="fr", batch_size=32):
        # :param language: language to load
        # :param batch_size: batch size
        self.batch_size = batch_size
        self.language = language
        print(f"Loading NLI dataset for {self.language}")
        self.dataset = NLIDataset(language)
        print("NLI dataset loaded")

    def get_dataloader(self, data_type="train") -> DataLoader:
        # :param data_type: type of data to load. Default to train because we want a large dataset
        # :return: DataLoader for the dataset
        dataloader = DataLoader(
            self.dataset.dataset[data_type], batch_size=self.batch_size, shuffle=True
        )
        return dataloader
