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

    def __init__(self, task, language="en", batch_size=32, sample_size=100, seed=42):
        super().__init__(task=task, language=language,
                         batch_size=batch_size, sample_size=sample_size, seed=seed)
        # get a random smaller sample from dataset
        sample_texts, sample_labels = self.get_random_sample(
            self, sample_size, seed=42)
        self.sampled_dataset = {
            "review_body": sample_texts, "stars": sample_labels}
        self.dataset = self.sampled_dataset

    def collate_fn(self, x):
        return [(self.prompt([row["review_body"]]), self.label_map(row["stars"])) for row in x]

    def get_random_sample(self, sample_size, seed=42):
        random.seed(seed)
        sample_indices = random.sample(
            range(len(self.dataset)), min(sample_size, len(self.dataset)))
        sample_texts = [self.dataset[i]["review_body"] for i in sample_indices]
        sample_labels = [self.dataset[i]["stars"] for i in sample_indices]
        return sample_texts, sample_labels
