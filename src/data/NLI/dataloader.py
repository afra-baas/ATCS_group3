
import random
from data.hf_dataloader import HFDataloader
from datetime import datetime


class NLIDataLoader(HFDataloader):
    data_name = "NLI"
    dataset_name = "xnli"

    def __init__(self, prompt_templates, language="en", task='SA', batch_size=32, sample_size=100, seed=42, data_type='train', use_oneshot=False, answer_type_ABC=False):
        super().__init__(prompt_templates, language=language, task=task,
                         batch_size=batch_size, sample_size=sample_size, seed=seed, data_type=data_type, use_oneshot=use_oneshot, answer_type_ABC=answer_type_ABC)

        start_time = datetime.now()
        use_neutral = True
        if use_neutral == False:
            self.dataset = self.filter_data()
            print('len dataset ', len(self.dataset))

        end_time = datetime.now()
        duration = end_time - start_time
        print(f'filter data Duraction: {duration}')

        start_time = datetime.now()
        self.dataset = self.get_random_sample(
            use_neutral=use_neutral, use_oneshot=use_oneshot)
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

    def get_random_sample(self, use_neutral=True, use_oneshot=False):
        # Gets a random sample from the dataset
        # :param sample_size: number of samples to get
        # :param seed: random seed
        random.seed(self.seed)
        self.entail_examples = [
            row for row in self.dataset if row["label"] == 0]
        if use_neutral:
            self.neutral_examples = [
                row for row in self.dataset if row["label"] == 1]
        self.contra_examples = [
            row for row in self.dataset if row["label"] == 2]

        if use_oneshot:
            print('*** using one shot approach')
            version = self.version
            # filename = f"list_indices_one_shot_3_{self.language}_{self.task}_{version}.py"
            module_name = f"one_shot.list_indices_one_shot_3_{self.language}_{self.task}_{version}"
            module = __import__(module_name)
            list_indices = getattr(module, "list_indices")
            one_shot_ent_ids, one_shot_neut_ids, one_shot_cont_ids = list_indices
        else:
            one_shot_ent_ids, one_shot_neut_ids, one_shot_cont_ids = [
                [], [], []]

        available_ent_sample_indices = [idx for idx in range(
            len(self.entail_examples)) if idx not in one_shot_ent_ids]
        available_neut_sample_indices = [idx for idx in range(
            len(self.neutral_examples)) if idx not in one_shot_neut_ids]
        available_cont_sample_indices = [idx for idx in range(
            len(self.contra_examples)) if idx not in one_shot_cont_ids]

        print('use_neutral is ', use_neutral)
        if use_neutral:
            self.ent_sample_indices = random.sample(
                available_ent_sample_indices, min(int(self.sample_size/3), len(self.entail_examples)))
            entail_examples = [self.entail_examples[i]
                               for i in self.ent_sample_indices]

            self.neut_sample_indices = random.sample(
                available_neut_sample_indices, min(int(self.sample_size/3), len(self.neutral_examples)))
            neutral_examples = [self.neutral_examples[i]
                                for i in self.neut_sample_indices]

            self.cont_sample_indices = random.sample(
                available_cont_sample_indices, min(int(self.sample_size/3), len(self.contra_examples)))
            contra_examples = [self.contra_examples[i]
                               for i in self.cont_sample_indices]

            print('len of entail_examples , neutral_examples, contra_examples: ',
                  len(entail_examples), len(neutral_examples), len(contra_examples))
            data = entail_examples + neutral_examples+contra_examples
        else:
            self.ent_sample_indices = random.sample(
                available_ent_sample_indices, min(int(self.sample_size/2), len(self.entail_examples)))
            entail_examples = [self.entail_examples[i]
                               for i in self.ent_sample_indices]

            self.cont_sample_indices = random.sample(
                available_cont_sample_indices, min(int(self.sample_size/2), len(self.contra_examples)))
            contra_examples = [self.contra_examples[i]
                               for i in self.cont_sample_indices]

            self.neut_sample_indices = []

            print('len we only take entail_examples , contra_examples: ',
                  len(entail_examples), len(contra_examples))
            data = entail_examples+contra_examples
        return data
