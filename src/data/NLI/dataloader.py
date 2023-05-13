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

    def __init__(self, language="en", batch_size=32, sample_size=100, seed=42):
        super().__init__(language=language,
                         batch_size=batch_size, sample_size=sample_size, seed=seed)
        # get a random smaller sample from dataset
        sample_pre, sample_hypo, sample_labels = self.get_random_sample(
            self, sample_size, seed=42)

        self.sampled_dataset = {
            "premise": sample_pre, "hypothesis": sample_hypo, "label": sample_labels}
        self.dataset = self.sampled_dataset

    def collate_fn(self, x):
        return [(self.prompt([row["premise"], row["hypothesis"]]), self.label_map(row["label"])) for row in x]

    def get_random_sample(self, sample_size, seed=42):
        random.seed(seed)
        sample_indices = random.sample(
            range(len(self.dataset)), min(sample_size, len(self.dataset)))
        sample_prem = [self.dataset[i]['premise'] for i in sample_indices]
        sample_hypo = [self.dataset[i]['hypothesis'] for i in sample_indices]
        sample_labels = [self.dataset[i]["label"] for i in sample_indices]
        return sample_prem, sample_hypo, sample_labels
