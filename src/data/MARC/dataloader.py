from torch.utils.data import DataLoader
from src.data.hf_dataloader import HFDataloader
from src.data.MARC.dataset import MARCDataset
import random

# Define the DataLoader class for NLI


class MARCDataLoader(HFDataloader):
    data_name = "MARC"
    language = "fr"
    supported_tasks = ["SA"]
    dataset_class = MARCDataset
    default_task = "SA"

    def collate_fn(self, x):

        return [(self.prompt([row["review_body"]]), self.label_map[int(row["stars"])]) for row in x]
