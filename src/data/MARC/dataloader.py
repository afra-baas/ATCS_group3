from torch.utils.data import DataLoader
from src.data.hf_dataloader import HFDataloader
from src.data.MARC.dataset import MARCDataset
from config import model, data


# Define the DataLoader class for NLI
class MARCDataLoader(HFDataloader):
    dataset_class = MARCDataset
    dataloader_name = "MARC"

    def collate_fn(self, x):
        return [(self.prompt([row["review_body"]]), self.label_to_meaning[int(row["stars"])]) for row in x]
