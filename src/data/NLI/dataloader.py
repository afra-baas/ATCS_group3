from src.data.NLI.dataset import NLIDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.data.hf_dataloader import HFDataloader

# Define the DataLoader class for NLI
class NLIDataLoader(HFDataloader):
    data_name = "MARC"
    language = "fr"
    supported_tasks = []
    dataset_class = NLIDataset

    def collate_fn(self, x):
        return [([row["premise"], row["hypothesis"]], row["label"]) for row in x]
