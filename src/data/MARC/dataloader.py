from torch.utils.data import DataLoader
from data.hf_dataloader import HFDataloader
from data.MARC.dataset import MARCDataset
import random


class MARCDataLoader(HFDataloader):
    data_name = "MARC"
    language = "en"
    supported_tasks = ["SA"]
    dataset_class = MARCDataset
    default_task = "SA"

    def __init__(self, prompt_type, prompt_id, language="en", task='SA', batch_size=32, sample_size=100, seed=42, data_type='train'):
        super().__init__(prompt_type, prompt_id, language=language, task=task,
                         batch_size=batch_size, sample_size=sample_size, seed=seed, data_type=data_type)
        # filter only 5 or 0 star results and reviews with <=40 tokens
        self.dataset = self.filter_data()
        print('len dataset ', len(self.dataset))

        self.dataset = self.get_random_sample()
        print('len dataset ', len(self.dataset))

    def collate_fn(self, x):
        batch = [(self.prompt([row["review_body"]], self.prompt_type, self.prompt_id),
                  self.label_map[row["stars"].item()]) for row in x]
        return zip(*batch)

    def filter_data(self):
        pos_reviews = [{"review_body": row["review_body"], "stars": row["stars"]}
                       for row in self.dataset if row["stars"] == 5 and len(row["review_body"].split()) <= 100]
        neg_reviews = [{"review_body": row["review_body"], "stars": row["stars"]}
                       for row in self.dataset if row["stars"] == 1 and len(row["review_body"].split()) <= 100]

        # Calculate average length of review_body for rows with stars equal to 1
        review_lengths = [len(row["review_body"])
                          for row in self.dataset if row["stars"] == 1]
        average_length = sum(review_lengths) / len(review_lengths)
        print("Average length of review_body for rows with 1 star:", average_length)
        review_lengths = [len(row["review_body"])
                          for row in self.dataset if row["stars"] == 5]
        average_length = sum(review_lengths) / len(review_lengths)
        print("Average length of review_body for rows with 5 star:", average_length)

        # balanced data, take the same lowest amount of both reviews
        lowest = min(len(pos_reviews), len(neg_reviews))
        print('len of lowest cat: ', lowest)
        pos_reviews = pos_reviews[:lowest]
        neg_reviews = neg_reviews[:lowest]
        data = pos_reviews + neg_reviews
        return data
