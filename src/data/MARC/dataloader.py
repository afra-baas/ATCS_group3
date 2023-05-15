from torch.utils.data import DataLoader
from data.hf_dataloader import HFDataloader
from data.MARC.dataset import MARCDataset
import random

# Define the DataLoader class for NLI


class MARCDataLoader(HFDataloader):
    data_name = "MARC"
    language = "en"
    supported_tasks = ["SA"]
    dataset_class = MARCDataset
    default_task = "SA"

    def collate_fn(self, x):
        # for row in x:
        #     print('row ', row)
        #     print(row["stars"].item())
        batch = [(self.prompt([row["review_body"]]),
                  self.label_map[row["stars"].item()]) for row in x]
        return zip(*batch)