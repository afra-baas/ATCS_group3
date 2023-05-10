from torch.utils.data import DataLoader
from src.data.hf_dataloader import HFDataloader
from src.data.MARC.dataset import MARCDataset

# Define the DataLoader class for NLI
class MARCDataLoader(HFDataloader):
    data_name = "MARC"
    language = "fr"
    supported_tasks = []
    dataset_class = MARCDataset

    def collate_fn(self, x):
        return [([row["review_body"]], row["stars"]) for row in x]
