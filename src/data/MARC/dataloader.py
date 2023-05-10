from torch.utils.data import DataLoader
from datasets import load_dataset
from config import data
from src.data.MARC.dataset import MARCDataset

# Define the DataLoader class for NLI
class MARCDataLoader:
    languages = {
        "English": "en",
        "German": "de",
        "Spanish": "es",
        "French": "fr",
        "Japanese": "ja",
        "Chinese": "zh",
    }
    MARC_DIR = data["MARC"]["path"]

    def __init__(
        self,
        data_path="",
        dataset_type="train",
        language="English",
        batch_size=32,
    ):
        # :param dataset: dataset to use
        # :param batch_size: batch size
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        try:
            self.language = self.languages[language]
        except:
            print("Language not in dataset")
            exit()
        if data_path != "":
            self.data_path = (
                "{marc_dir}/dataset_{langugage}_{dataset_type}.json".format(
                    marc_dir=self.MARC_DIR,
                    language=self.language,
                    dataset_type=self.dataset_type,
                )
            )
        print("Loading dataset from {}".format(self.data_path))
        self.dataset = MARCDataset(self.data_path, self.tokenizer)
        print("MARC dataset loaded")

    def get_dataloader(self):
        # :return: DataLoader for the dataset
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
