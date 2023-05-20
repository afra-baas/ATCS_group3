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

    def __init__(self, prompt_type, prompt_id, language="en", task='SA', batch_size=32, sample_size=100, seed=42, data_type='train'):
        # :param language: language to load
        # :param batch_size: batch size
        self.batch_size = batch_size
        self.language = language
        self.task = task
        print(f"Loading {self.data_name} dataset for {self.language}")
        self.dataset = self.dataset_class(language).dataset[data_type]
        print(f"{self.data_name} dataset loaded")
        task = self.default_task
        if task not in self.supported_tasks:
            print(f"Task {task} not supported for {self.data_name}")
            raise ValueError
        try:
            self.task_config = task_config["SUPPORTED_TASKS"][self.language][task]
        except KeyError:
            print(f"Task {task} not supported")
            raise KeyError
        self.prompt = self.task_config["prompt_class"](
            self.language, self.task)
        self.label_map = self.task_config["label_map"]
        self.possible_answers = self.task_config["possible_answers"]

        self.prompt_type = prompt_type
        self.prompt_id = prompt_id
        self.seed = seed
        self.sample_size = sample_size

    def __iter__(self):
        # :return: DataLoader for the dataset with shape (batch_size, 2). The first element of the tuple is a list of sentences, the second is an int representing the label
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        return iter(dataloader)

    def collate_fn(self, x):
        # To be implemented in child class
        raise NotImplementedError

    # def get_random_sample(self):
    #     # Gets a random sample from the dataset
    #     # :param sample_size: number of samples to get
    #     # :param seed: random seed
    #     random.seed(self.seed)
    #     sample_indices = random.sample(
    #         range(len(self.dataset)), min(self.sample_size, len(self.dataset)))
    #     dataset = [self.dataset[i] for i in sample_indices]
    #     return dataset
