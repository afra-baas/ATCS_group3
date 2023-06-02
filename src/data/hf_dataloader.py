from torch.utils.data import DataLoader
from config import task_config
import random
from datasets import load_dataset
from datetime import datetime


class HFDataloader:
    """
    Abstract class for dataloader
    """

    def __init__(self, prompt_templates, language="en", task='SA', batch_size=32, sample_size=200, seed=42, data_type='train', use_oneshot=False):
        # :param language: language to load
        # :param batch_size: batch size
        self.seed = seed
        self.batch_size = batch_size
        self.language = language
        self.task = task
        self.sample_size = sample_size
        self.version = 3

        start_time = datetime.now()
        print(f"Using {self.data_name} dataset for {self.language}")
        self.dataset = load_dataset(
            self.dataset_name, self.language).with_format("torch")[data_type]

        end_time = datetime.now()
        duration = end_time - start_time
        print(f'loading model Duration: {duration}')

        try:
            self.task_config = task_config["SUPPORTED_TASKS"][self.language][task]
        except KeyError:
            print(f"Task {task} not supported")
            raise KeyError

        self.prompt = self.task_config["prompt_class"](
            self.language, self.task, prompt_templates)
        self.label_map = self.task_config["label_map"]
        self.possible_answers = self.task_config["possible_answers"]

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
