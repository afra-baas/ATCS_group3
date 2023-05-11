from torch.utils.data import DataLoader
from config import task_config


class HFDataloader:
    """
    Abstract class for dataloader
    """

    data_name = ""
    language = ""
    supported_tasks = ["Empty"]
    dataset_class = None
    default_task = "Empty"

    def __init__(self, language="fr", task="", batch_size=32):
        # :param language: language to load
        # :param batch_size: batch size
        self.batch_size = batch_size
        self.language = language
        print(f"Loading {self.data_name} dataset for {self.language}")
        self.dataset = self.dataset_class(language)
        print(f"{self.data_name} dataset loaded")
        if not task:
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

    def get_dataloader(self, data_type="train") -> DataLoader:
        # :return: DataLoader for the dataset with shape (batch_size, 2). The first element of the tuple is a list of sentences, the second is an int representing the label
        dataloader = DataLoader(
            self.dataset.dataset[data_type],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def collate_fn(self, x):
        # To be implemented in child class
        raise NotImplementedError
