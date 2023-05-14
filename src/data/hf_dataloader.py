from torch.utils.data import DataLoader
# from ATCS_group3.src.config import task_config
from config import task_config
import random

class HFDataloader:
    """
    Abstract class for dataloader
    """

    data_name = ""
    language = ""
    supported_tasks = ["Empty"]
    dataset_class = None
    default_task = "Empty"

    def __init__(self, language="en", batch_size=32, sample_size=100, seed=42):
        # :param language: language to load
        # :param batch_size: batch size
        self.batch_size = batch_size
        self.language = language
        print(f"Loading {self.data_name} dataset for {self.language}")
        self.dataset = self.dataset_class(language)#.dataset
        print(f"{self.data_name} dataset loaded")
        task = self.default_task
        if task not in self.supported_tasks:
            print(f"Task {task} not supported for {self.data_name}")
            raise ValueError
        try:
            self.task_config = task_config["SUPPORTED_TASKS"][task]
        except KeyError:
            print(f"Task {task} not supported")
            raise KeyError
        self.prompt = self.task_config["prompt_class"]()
        self.label_map = self.task_config["label_map"]
        self.possible_answers = self.task_config["possible_answers"]
        self.dataset = self.get_random_sample(self.dataset, sample_size, seed)

    def get_dataloader(self, data_type="train") -> DataLoader:
        # :return: DataLoader for the dataset with shape (batch_size, 2). The first element of the tuple is a list of sentences, the second is an int representing the label
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def collate_fn(self, x):
        # To be implemented in child class
        raise NotImplementedError

    def get_random_sample(self, sample_size: int, seed: int = 42):
        # Gets a random sample from the dataset
        # :param sample_size: number of samples to get
        # :param seed: random seed
        random.seed(seed)
        sample_indices = random.sample(
            range(len(self.dataset)), min(sample_size, len(self.dataset)))
        return [self.dataset[i] for i in sample_indices]
