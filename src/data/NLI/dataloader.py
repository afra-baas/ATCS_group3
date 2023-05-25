
import random
from data.hf_dataloader import HFDataloader
from datetime import datetime


class NLIDataLoader(HFDataloader):
    data_name = "NLI"
    dataset_name = "xnli"

    def __init__(self, language="en", task='SA', batch_size=32, sample_size=100, seed=42, data_type='train'):
        super().__init__(language=language, task=task,
                         batch_size=batch_size, sample_size=sample_size, seed=seed, data_type=data_type)

        start_time = datetime.now()
        self.dataset = self.filter_data()
        print('len dataset ', len(self.dataset))

        end_time = datetime.now()
        duration = end_time - start_time
        print(f'filter data Duraction: {duration}')

        start_time = datetime.now()
        self.dataset = self.get_random_sample()
        print('len dataset ', len(self.dataset))

        end_time = datetime.now()
        duration = end_time - start_time
        print(f'random sample Duraction: {duration}')

    def collate_fn(self, x):
        batch = [([row["premise"], row["hypothesis"]], row["label"].item())
                 for row in x]
        return zip(*batch)

    def filter_data(self):
        # filter out neutral
        reviews = [row for row in self.dataset if row["label"] != 1]
        return reviews

    def get_random_sample(self, use_neutral=True):
        # Gets a random sample from the dataset
        # :param sample_size: number of samples to get
        # :param seed: random seed
        random.seed(self.seed)

        self.entail_examples = [
            row for row in self.dataset if row["label"] == 0]
        if use_neutral:
            self.neutral_examples = [
                row for row in self.dataset if row["label"] == 1]
            print(' the amount of neutral are in the whole dataset: ',
                  len(self.neutral_examples))
        self.contra_examples = [
            row for row in self.dataset if row["label"] == 2]

        print('use_neutral is ', use_neutral)
        if use_neutral:
            sample_indices = random.sample(
                range(len(self.entail_examples)), min(int(self.sample_size/3), len(self.entail_examples)))
            entail_examples = [self.entail_examples[i] for i in sample_indices]

            sample_indices = random.sample(
                range(len(self.neutral_examples)), min(int(self.sample_size/3), len(self.neutral_examples)))
            neutral_examples = [self.neutral_examples[i]
                                for i in sample_indices]

            sample_indices = random.sample(
                range(len(self.contra_examples)), min(int(self.sample_size/3), len(self.contra_examples)))
            contra_examples = [self.contra_examples[i] for i in sample_indices]

            print('len of entail_examples , neutral_examples, contra_examples: ',
                  len(entail_examples), len(neutral_examples), len(contra_examples))
            data = entail_examples + neutral_examples+contra_examples
        else:
            sample_indices = random.sample(
                range(len(self.entail_examples)), min(int(self.sample_size/2), len(self.entail_examples)))
            entail_examples = [self.entail_examples[i] for i in sample_indices]

            sample_indices = random.sample(
                range(len(self.contra_examples)), min(int(self.sample_size/2), len(self.contra_examples)))
            contra_examples = [self.contra_examples[i] for i in sample_indices]

            print('len we only take entail_examples , contra_examples: ',
                  len(entail_examples), len(contra_examples))
            data = entail_examples+contra_examples
        return data
