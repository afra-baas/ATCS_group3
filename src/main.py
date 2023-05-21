import argparse
# from config import data, model, task_config
from models.model import Model
from data.MARC.dataloader import MARCDataLoader
from data.NLI.dataloader import NLIDataLoader
from eval import evaluate
import torch
from datetime import datetime
import os
import pickle
from visualize import get_acc_plot, get_box_plot
import collections


def makehash():
    return collections.defaultdict(makehash)


def default_to_regular(d):
    if isinstance(d, collections.defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def get_prompt_acc(seed, lang, LM, task, prompt_type, prompt_id, sample_size, batch_size):

    # if this function is called directly, instead of through the pipeline
    if type(LM) == str:
        LM = Model(LM)

    print("-----------", seed, lang, LM.model_name, task, prompt_type,
          prompt_id, sample_size, batch_size, '--------------')
    logits_dict_for_prompt = {}
    start_time = datetime.now()
    if task == 'SA':
        train_dataloader = MARCDataLoader(prompt_type, prompt_id, language=lang, task=task,
                                          sample_size=sample_size, batch_size=batch_size, seed=seed)
    elif task == 'NLI':
        train_dataloader = NLIDataLoader(prompt_type, prompt_id, language=lang, task=task,
                                         sample_size=sample_size, batch_size=batch_size, seed=seed)
    else:
        print('This task evaluation is not implemented')

    answers_probs_all = []
    pred_answer_all = []
    mapped_labels_all = []
    i = 0
    possible_answers = train_dataloader.possible_answers
    sample_id = 0
    for batch in train_dataloader:
        print(
            f'Batch: {i} , batch size: {batch_size}, sample_size: {sample_size}')
        prompts, mapped_labels = batch

        # Classification
        answers_probs_batch, pred_answer_batch = LM(
            prompts, possible_answers, language=lang)

        # save logits per batch (*i+1 so sent_id is from 0 to sent_id*batch_size)
        # NOTE: sample_id, because we want each sentence in the sample sixe the an id (per sen in each batch)

        if task == 'SA':
            for sent_id, probs in enumerate(answers_probs_batch):
                logits_dict_for_prompt[sample_id] = {
                    'yes': probs[0].detach().item(),
                    'no': probs[1].detach().item(),
                    'diff': abs(probs[0] - probs[1]).detach().item(),
                    'pred': pred_answer_batch[sent_id],
                    'true': mapped_labels[sent_id]
                }
                sample_id += 1

        else:
            for sent_id, probs in enumerate(answers_probs_batch):
                sorted_list = sorted(
                    [probs[0], probs[1], probs[2]], reverse=True)
                logits_dict_for_prompt[sample_id] = {
                    'yes': probs[0].detach().item(),
                    'no': probs[1].detach().item(),
                    'maybe': probs[2].detach().item(),
                    'diff': abs(sorted_list[0]-sorted_list[1]).detach().item(),
                    'pred': pred_answer_batch[sent_id],
                    'true': mapped_labels[sent_id]
                }
                sample_id += 1

        answers_probs_all.extend(answers_probs_batch)
        pred_answer_all.extend(pred_answer_batch)
        mapped_labels_all.extend(mapped_labels)
        i += 1

    # Evaluation
    acc = evaluate(pred_answer_all, mapped_labels_all)
    print('acc: ', acc)

    # also add the acc after all sentences to needing another nested dict
    logits_dict_for_prompt['acc'] = acc

    end_time = datetime.now()
    duration = end_time - start_time
    print(
        f"Time taken to execute the {lang} {task} task with prompt type {prompt_type}, variation {prompt_id} and batchsize {batch_size}: {duration}")

    return logits_dict_for_prompt


def pipeline(seeds, languages, LM_models, tasks, prompt_types, batch_size, sample_size, num_prompts, file_path=f'./ATCS_group3/saved_outputs/logits_dict.pickle'):

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # logits_dict structure: specify seed, lang, model, task, prompt_type, prompt_id (,sentence_num, answer)
    logits_dict = makehash()
    for seed in seeds:
        for lang in languages:
            for LM_model in LM_models:
                # Initilize model
                LM = Model(LM_model)
                for task in tasks:
                    # with this we look at different charactistics in a prompt
                    for prompt_type in prompt_types:
                        # with this we will look at variability in the prompt type
                        for prompt_id in range(num_prompts):

                            logits_dict_for_prompt = get_prompt_acc(
                                seed, lang, LM, task, prompt_type, prompt_id, sample_size, batch_size)
                            path = [seed, lang, LM_model, task,
                                    prompt_type, f'prompt_id_{prompt_id}']
                            print('path', path)
                            logits_dict[seed][lang][LM_model][task][prompt_type][
                                f'prompt_id_{prompt_id}'] = logits_dict_for_prompt

                        # Open the file in binary mode and save the dictionary
                        with open(file_path, 'wb') as file:
                            pickle.dump(default_to_regular(logits_dict), file)
                        # Save the code snippet to the text file
                        with open(file_path.replace('.pickle', '.txt'), 'w+') as file:
                            for key, value in default_to_regular(logits_dict).items():
                                file.write(f'{key}: {value}\n')
                        print(
                            f"Dictionary saved to '{file_path}' as a pickle file.")


if __name__ == "__main__":

    models = ['bloom', 'bloomz']  # 'flan', 'llama' , 'alpaca']
    # models = ['llama']
    tasks = ['NLI', 'SA']
    prompt_types = ['active', 'passive', 'auxiliary', 'modal', 'rare_synonyms']
    # prompt_types = ['active', 'passive']
    # languages = ['en', 'de']
    languages = ['en']
    # seeds = ['42', '33', '50']
    seeds = ['42']

    batch_size = 16
    sample_size = 200
    num_prompts = 6

    # MAKE sure the change this if you dont want to overwrite previous results
    version = 5
    for seed in seeds:
        for lang in languages:
            file_path = f'./ATCS_group3/saved_outputs/logits_dict_seed_{seed}_lang_{lang}_v{version}.pickle'

            print('****Start Time:', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            start_time = datetime.now()

            pipeline(seeds, languages, models, tasks, prompt_types,
                     batch_size, sample_size, num_prompts, file_path=file_path)

            end_time = datetime.now()
            duration = end_time - start_time
            print('****End Time:', end_time, f'Duraction: {duration}')
