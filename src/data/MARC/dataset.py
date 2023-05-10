from src.data.hf_dataset import HFDataset


class MARCDataset(HFDataset):
    language = "fr"
    dataset_name = "amazon_reviews_multi"

