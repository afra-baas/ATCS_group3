from config import data
from src.data.NLI.dataset import NLIDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.data.hf_dataloader import HFDataloader

# Define the DataLoader class for NLI
class NLIDataLoader(HFDataloader):
    dataloader_name = "NLI"
    dataset_class = NLIDataset
    
    def collate_fn(self, x):
        return [(self.prompt([row["premise"], row["hypothesis"]]), self.label_to_meaning[int(row["label"])]) for row in x]
