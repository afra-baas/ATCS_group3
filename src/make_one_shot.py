from prompts.prompt import Prompt
from typing import List
from prompt_structure_without_one_shot import prompt_templates
from datetime import datetime
from data.hf_dataloader import HFDataloader
from models.model import Model
from data.MARC.dataloader import MARCDataLoader
from data.NLI.dataloader import NLIDataLoader
import os


class SAPrompt(Prompt):

    def __call__(self, sentences: List[str], prompt_type, prompt_id) -> str:
        # :param sentences: a list with all the input sentences ## ??
        # :return: a string transformed to the desired prompt.

        template = self.dict_sa_prompt[prompt_type][prompt_id]
        content = sentences[0]
        prompt = template.format(content=content)
        return prompt


class NLIPrompt(Prompt):

    def __call__(self, sentences: List[str], prompt_type, prompt_id) -> str:
        # :param sentences: a list with all the input sentences ## ??
        # :return: a string transformed to the desired prompt.

        template = self.dict_sa_prompt[prompt_type][prompt_id]
        premise = sentences[0]
        hypothesis = sentences[1]
        prompt = template.format(premise=premise, hypothesis=hypothesis)
        return prompt


if __name__ == "__main__":
    models = ['bloom', 'bloomz', 'flan', 'llama', 't0']
    tasks = ['SA', 'NLI']
    prompt_types = ['active', 'passive', 'auxiliary',
                    'modal', 'common', 'rare_synonyms', 'identical_modal']
    languages = ['en', 'de', 'fr']
    seeds = ['3']

    batch_size = 10
    sample_size = 210

    # MAKE sure the change this if you dont want to overwrite previous results
    version = 1

    print('****Start Time:', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    start_time = datetime.now()

    for seed in seeds:
        for lang in languages:
            for task in tasks:
                start_time = datetime.now()
                if task == 'SA':
                    train_dataloader = MARCDataLoader(language=lang, task=task,
                                                      sample_size=sample_size, batch_size=batch_size, seed=seed, data_type='train')
                else:
                    train_dataloader = NLIDataLoader(language=lang, task=task,
                                                     sample_size=sample_size, batch_size=batch_size, seed=seed, data_type='train')
                for LM_model in models:
                    # Initilize model
                    LM = Model(LM_model, train_dataloader.possible_answers)
                    for prompt_type in prompt_types:
                        num_prompts = len(
                            prompt_templates[lang][task][prompt_type])
                        print(
                            f'prompt_type {prompt_type} has {num_prompts} prompts in it')
                        new_prompts = []
                        for prompt_id in range(num_prompts):
                            og_prompt = prompt_templates[lang][task][prompt_type][prompt_id]
                            for i, batch in enumerate(train_dataloader):
                                sentences, labels = batch
                                # for j, sentence in enumerate(sentences):
                            mapped_label = train_dataloader.label_map[labels[prompt_id]]
                            one_shot = " ".join(train_dataloader.prompt(
                                sentences[prompt_id], prompt_type, prompt_id), mapped_label)
                            new_prompts.append(
                                "\n \n".join(one_shot, og_prompt))
                            print(len(new_prompts))
                            if prompt_type == 'active' and prompt_id == 0:
                                print(new_prompts[0])

                        prompt_templates[lang][task][prompt_type] = num_prompts

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(
        './ATCS_group3/src/prompt_structure_one_shot{version}.py'), exist_ok=True)

    # Convert the dictionary to a string representation
    data_str = str(prompt_templates)
    # Write the string representation to a Python file
    with open(f"./ATCS_group3/src/prompt_structure_one_shot{version}.py", "w+") as file:
        file.write("data = " + data_str)

    end_time = datetime.now()
    duration = end_time - start_time
    print('****End Time:', end_time, f'Duration: {duration}')
