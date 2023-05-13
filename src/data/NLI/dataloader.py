from src.data.NLI.dataset import NLIDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
import random

from src.data.hf_dataloader import HFDataloader


# Define the DataLoader class for NLI
class NLIDataLoader(HFDataloader):
    data_name = "NLI"
    language = "en"
    supported_tasks = ["NLI", "Empty"]
    dataset_class = NLIDataset
    default_task = "NLI"

    def collate_fn(self, x):
        return [(self.prompt([row["premise"], row["hypothesis"]]), self.label_map[int(row["label"])]) for row in x]
