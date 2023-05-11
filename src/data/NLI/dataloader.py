from src.data.NLI.dataset import NLIDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.data.hf_dataloader import HFDataloader

# Define the DataLoader class for NLI
class NLIDataLoader(HFDataloader):
    data_name = "NLI"
    language = "fr"
    supported_tasks = ["NLI", "Empty"]
    dataset_class = NLIDataset
    default_task = "NLI"

    def collate_fn(self, x):
        return [(self.prompt([row["premise"], row["hypothesis"]]), row["label"]) for row in x]
