from torch.utils.data import DataLoader


class HFDataloader:
    """
    Abstract class for dataloader
    """

    data_name = ""
    language = ""
    supported_tasks = []
    dataset_class = None

    def __init__(self, language="fr", batch_size=32):
        # :param language: language to load
        # :param batch_size: batch size
        self.batch_size = batch_size
        self.language = language
        print(f"Loading {self.data_name} dataset for {self.language}")
        self.dataset = self.dataset_class(language)
        print(f"{self.data_name} dataset loaded")

    def get_dataloader(self, data_type="train") -> DataLoader:
        # :return: DataLoader for the dataset
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
