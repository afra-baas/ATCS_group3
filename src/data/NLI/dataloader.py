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

    def get_random_sample(self):
        # Gets a random sample from the dataset
        # :param sample_size: number of samples to get
        # :param seed: random seed
        random.seed(self.seed)

        self.entail_examples = [self.dataset[row]
                                for row in self.dataset if row["label"] == 0]
        self.neutral_examples = [self.dataset[row]
                                 for row in self.dataset if row["label"] == 1]
        self.contra_examples = [self.dataset[row]
                                for row in self.dataset if row["label"] == 2]

        sample_indices = random.sample(
            range(len(self.entail_examples)), min(int(self.sample_size/3), len(self.entail_examples)))
        entail_examples = [self.entail_examples[i] for i in sample_indices]

        sample_indices = random.sample(
            range(len(self.neutral_examples)), min(int(self.sample_size/3), len(self.neutral_examples)))
        neutral_examples = [self.neutral_examples[i] for i in sample_indices]

        sample_indices = random.sample(
            range(len(self.contra_examples)), min(int(self.sample_size/3), len(self.contra_examples)))
        contra_examples = [self.contra_examples[i] for i in sample_indices]

        print('len of entail_examples , neutral_examples, contra_examples: ',
              len(entail_examples), len(neutral_examples), len(contra_examples))
        data = entail_examples + neutral_examples+contra_examples
        return data
