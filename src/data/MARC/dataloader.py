from data.hf_dataloader import HFDataloader
import random


class MARCDataLoader(HFDataloader):
    data_name = "MARC"
    dataset_name = "amazon_reviews_multi"

    def __init__(self, language="en", task='SA', batch_size=32, sample_size=200, seed=42, data_type='train', use_oneshot=False):
        super().__init__(language=language, task=task,
                         batch_size=batch_size, sample_size=sample_size, seed=seed, data_type=data_type, use_oneshot=use_oneshot)
        # filter only 5 or 0 star results and reviews with <=40 tokens
        self.dataset = self.filter_data()
        print('len dataset ', len(self.dataset))

        self.dataset = self.get_random_sample(use_oneshot=use_oneshot)
        print('len dataset ', len(self.dataset))

    def collate_fn(self, x):
        batch = [([row["review_body"]], row["stars"].item()) for row in x]
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
        self.pos_reviews = pos_reviews[:lowest]
        self.neg_reviews = neg_reviews[:lowest]
        data = self.pos_reviews + self.neg_reviews
        print('len of pos_reviews, neg_reviews: ',
              len(self.pos_reviews), len(self.neg_reviews))
        return data

    def get_random_sample(self, use_oneshot=False):
        # Gets a random sample from the dataset
        # :param sample_size: number of samples to get
        # :param seed: random seed
        random.seed(self.seed)

        if use_oneshot:
            print('*** using one shot approach')
            version = self.version
            # filename = f"list_indices_one_shot_3_{self.language}_{self.task}_{version}.py"
            module_name = f"one_shot.list_indices_one_shot_3_{self.language}_{self.task}_{version}"
            module = __import__(module_name)
            list_indices = getattr(module, "list_indices")
            one_shot_pos_ids, one_shot_neg_ids = list_indices
        else:
            one_shot_pos_ids, one_shot_neg_ids = [[], []]

        available_pos_sample_indices = [idx for idx in range(
            len(self.pos_reviews)) if idx not in one_shot_pos_ids]
        available_neg_sample_indices = [idx for idx in range(
            len(self.neg_reviews)) if idx not in one_shot_neg_ids]

        self.pos_sample_indices = random.sample(
            available_pos_sample_indices, min(int(self.sample_size/2), len(self.pos_reviews)))
        pos_reviews = [self.pos_reviews[i] for i in self.pos_sample_indices]

        self.neg_sample_indices = random.sample(
            available_neg_sample_indices, min(int(self.sample_size/2), len(self.neg_reviews)))
        neg_reviews = [self.neg_reviews[i] for i in self.neg_sample_indices]
        data = pos_reviews + neg_reviews
        return data
