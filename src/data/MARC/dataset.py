from data.hf_dataset import HFDataset


class MARCDataset(HFDataset):
    language = "en"
    dataset_name = "amazon_reviews_multi"
