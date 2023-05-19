import numpy
from datasets import load_dataset
from torch.utils.data import Subset


class HFDataset:
    language = ""
    dataset_name = ""

    def __init__(self, language="en"):
        # :param dataset: dataset to use
        # :param tokenizer: tokenizer to use
        self.language = language
        self.dataset = load_dataset(self.dataset_name, self.language).with_format(
            "torch"
        )
