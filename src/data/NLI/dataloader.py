from data.NLI.dataset import NLIDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
import random

from data.hf_dataloader import HFDataloader


# Define the DataLoader class for NLI
class NLIDataLoader(HFDataloader):
    data_name = "NLI"
    language = "en"
    supported_tasks = ["NLI"]
    dataset_class = NLIDataset
    default_task = "NLI"

    def __init__(self, prompt_type, prompt_id, language="en", task='SA', batch_size=32, sample_size=100, seed=42, data_type='train'):
        super().__init__(prompt_type, prompt_id, language=language, task=task,
                         batch_size=batch_size, sample_size=sample_size, seed=seed, data_type=data_type)

        self.dataset = self.get_random_sample()
        print('len dataset ', len(self.dataset))

    def collate_fn(self, x):
        batch = [(self.prompt([row["premise"], row["hypothesis"]], self.prompt_type, self.prompt_id),
                  self.label_map[row["label"].item()]) for row in x]
        return zip(*batch)
