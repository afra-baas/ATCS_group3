from datetime import datetime
from data.MARC.dataloader import MARCDataLoader
from data.NLI.dataloader import NLIDataLoader
import os


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
    version = 3

    # specify here which prompt structure you want to import
    module_name = f"prompts.templates.prompt_without"
    module = __import__(module_name, fromlist=["prompt_templates"])
    prompt_templates = getattr(module, "prompt_templates")

    print('****Start Time:', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    start_time1 = datetime.now()

    for seed in seeds:
        for lang in languages:
            for task in tasks:
                start_time = datetime.now()
                if task == 'SA':
                    train_dataloader = MARCDataLoader(language=lang, task=task,
                                                      sample_size=sample_size, batch_size=batch_size, seed=seed, data_type='train', use_oneshot=False)
                    list_indices = [
                        train_dataloader.pos_sample_indices, train_dataloader.neg_sample_indices]
                else:
                    train_dataloader = NLIDataLoader(language=lang, task=task,
                                                     sample_size=sample_size, batch_size=batch_size, seed=seed, data_type='train', use_oneshot=False)
                    list_indices = [train_dataloader.ent_sample_indices,
                                    train_dataloader.neut_sample_indices, train_dataloader.cont_sample_indices]

                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(
                    f"./ATCS_group3/one_shot_files/list_indices_one_shot_{seed}_{lang}_{task}_{version}.py"), exist_ok=True)

                # Write the string representation to a Python file
                with open(f"./ATCS_group3/one_shot_files/list_indices_one_shot_{seed}_{lang}_{task}_{version}.py", "w+") as file:
                    file.write("list_indices = " + str(list_indices))

                for prompt_type in prompt_types:
                    print(f'path: {lang}{task}{prompt_type}')
                    num_prompts = len(
                        prompt_templates[lang][task][prompt_type])
                    print(
                        f'prompt_type {prompt_type} has {num_prompts} prompts in it')
                    new_prompts = []
                    for prompt_id in range(num_prompts):
                        og_prompt = prompt_templates[lang][task][prompt_type][prompt_id]
                        for i, batch in enumerate(train_dataloader):
                            sentences, labels = batch
                            mapped_label = train_dataloader.label_map[labels[prompt_id]]
                            example = train_dataloader.prompt(
                                sentences[prompt_id], prompt_type, prompt_id)
                            one_shot = " ".join([example, mapped_label])
                            new_prompts.append(
                                "\n \n".join([one_shot, og_prompt]))

                        print(len(new_prompts))
                        if prompt_type == 'active' and prompt_id == 0:
                            print(new_prompts[0])

                    prompt_templates[lang][task][prompt_type] = new_prompts

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(
        f'./ATCS_group3/one_shot_files/prompt_structure_one_shot{version}.py'), exist_ok=True)

    # Write the string representation to a Python file
    with open(f"./ATCS_group3/one_shot_files/prompt_structure_one_shot{version}.py", "w+") as file:
        file.write("prompt_templates = " + str(prompt_templates))

    end_time = datetime.now()
    duration = end_time - start_time1
    print('****End Time:', end_time, f'Duration: {duration}')
