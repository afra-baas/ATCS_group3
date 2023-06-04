import argparse
from models.model import Model
from data.MARC.dataloader import MARCDataLoader
from data.NLI.dataloader import NLIDataLoader

from eval import evaluate
from datetime import datetime
import os
import pickle
import collections


def makehash():
    return collections.defaultdict(makehash)


def default_to_regular(d):
    if isinstance(d, collections.defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def get_prompt_acc(seed, train_dataloader, lang, LM, task, prompt_type, prompt_id, sample_size, batch_size):

    # if this function is called directly, instead of through the pipeline
    if type(LM) == str:
        LM = Model(LM)

    print("-----------", seed, lang, LM.model_name, task, prompt_type,
          prompt_id, sample_size, batch_size, '--------------')
    start_time1 = datetime.now()

    logits_dict_for_prompt = {}
    answers_probs_all = []
    pred_answer_all = []
    mapped_labels_all = []
    sample_id = 0
    for i, batch in enumerate(train_dataloader):

        sentences, labels = batch
        prompts = [train_dataloader.prompt(
            sentence, prompt_type, prompt_id) for sentence in sentences]
        mapped_labels = [train_dataloader.label_map[label] for label in labels]

        start_time = datetime.now()
        # Classification
        print('len prompts', len(prompts))
        answers_probs_batch, pred_answer_batch = LM(prompts)

        end_time = datetime.now()
        duration = end_time - start_time
        print(
            f' Batch: {i} of {prompt_type} classification Duration: {duration}')

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

    # Evaluation
    acc = evaluate(pred_answer_all, mapped_labels_all)
    print('acc: ', acc)

    # also add the acc after all sentences to needing another nested dict
    logits_dict_for_prompt['acc'] = acc

    end_time = datetime.now()
    duration = end_time - start_time1
    print(
        f"Time taken to execute the {lang} {task} task with prompt type {prompt_type}, variation {prompt_id} and batchsize {batch_size}: {duration}")

    return logits_dict_for_prompt


def pipeline(seed, lang, LM_models, tasks, prompt_types, prompt_templates, batch_size, sample_size, answer_type_ABC, data_type='train', use_oneshot=False, file_path=f'./ATCS_group3/saved_outputs/logits_dict.pickle'):

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # logits_dict structure: specify seed, lang, model, task, prompt_type, prompt_id (,sentence_num, answer)
    logits_dict = makehash()
    for task in tasks:
        start_time = datetime.now()
        if task == 'SA':
            train_dataloader = MARCDataLoader(prompt_templates, language=lang, task=task,
                                              sample_size=sample_size, batch_size=batch_size, seed=seed, data_type=data_type, use_oneshot=use_oneshot, answer_type_ABC=answer_type_ABC)
        else:
            train_dataloader = NLIDataLoader(prompt_templates, language=lang, task=task,
                                             sample_size=sample_size, batch_size=batch_size, seed=seed, data_type=data_type, use_oneshot=use_oneshot, answer_type_ABC=answer_type_ABC)
        # TO DO:
        # saved the sampled sentences, in a dict?
        end_time = datetime.now()
        duration = end_time - start_time
        print(f'create dataloader Duration: {duration}')
        for LM_model in LM_models:
            start_time = datetime.now()
            # Initilize model
            LM = Model(LM_model, train_dataloader.possible_answers)
            end_time = datetime.now()
            duration = end_time - start_time
            print(f'load model Duration: {duration}')
            # with this we look at different charactistics in a prompt
            for prompt_type in prompt_types:
                # with this we will look at variability in the prompt type
                num_prompts = len(prompt_templates[lang][task][prompt_type])
                print(
                    f'prompt_type {prompt_type} has {num_prompts} prompts in it')
                for prompt_id in range(num_prompts):

                    # the the instruction bias, to scale normalize
                    answers_probs_batch, pred_answer_batch = LM(
                        train_dataloader.prompt.dict_sa_prompt[prompt_type][prompt_id])
                    scale = answers_probs_batch[0][0] / \
                        answers_probs_batch[0][1]  # yes/ no
                    LM.scale = scale

                    logits_dict_for_prompt = get_prompt_acc(
                        seed, train_dataloader, lang, LM, task, prompt_type, prompt_id, sample_size, batch_size)
                    path = [seed, lang, LM_model, task,
                            prompt_type, f'prompt_id_{prompt_id}']
                    print('path', path)
                    logits_dict[seed][lang][LM_model][task][prompt_type][
                        f'prompt_id_{prompt_id}'] = logits_dict_for_prompt

                # Open the file in binary mode and save the dictionary
                with open(file_path, 'wb') as file:
                    pickle.dump(default_to_regular(logits_dict), file)
                print(
                    f"Dictionary saved to '{file_path}' as a pickle file.")
                print()


if __name__ == "__main__":

    # models = ['bloom', 'bloomz', 'flan', 'llama', 't0', 't5']
    # tasks = ['SA', 'NLI']
    # prompt_types = ['active', 'passive', 'auxiliary',
    #                 'modal', 'common', 'rare_synonyms', 'identical_modal']
    # languages = ['en', 'de', 'fr']
    # seeds = ['33', '50'] #42

    # batch_size = 16
    # sample_size = 200

    # # MAKE sure the change this if you dont want to overwrite previous results
    # version = 96

    # # specify here which prompt structure you want to import
    # module_name = f"prompts.templates.prompt_structure_ABC_maybe"
    # module = __import__(module_name, fromlist=["prompt_templates"])
    # prompt_templates = getattr(module, "prompt_templates")

    # # specify if you want the probabilities on ABC (then = True) or yes no maybe
    # answer_type_ABC = True

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--models', nargs='+', help='List of models',
                        default=['bloom', 'bloomz', 'flan', 'llama', 't0'])
    parser.add_argument('--tasks', nargs='+',
                        help='List of tasks', default=['SA', 'NLI'])
    # parser.add_argument('--prompt_types', nargs='+', help='List of prompt types')
    parser.add_argument('--languages', nargs='+',
                        help='List of languages', default=['en', 'de', 'fr'])
    parser.add_argument('--seeds', nargs='+',
                        help='List of seeds', default=['42', '33', '50'])
    # parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--sample_size', type=int,
                        help='Sample size', default=200)
    parser.add_argument('--version', type=int,
                        help='Run version (to avoid previous output being overwritten)')
    parser.add_argument('--prompt_structure_name', type=str,
                        help='Name of prompt_structure file you want to use', default='prompt_structure_ABC_maybe')
    parser.add_argument('--answer_type_ABC', type=bool,
                        help='Answer type (True for ABC)', default=True)

    args = parser.parse_args()

    models = args.models
    tasks = args.tasks
    # prompt_types = args.prompt_types
    prompt_types = ['active', 'passive', 'auxiliary',
                    'modal', 'common', 'rare_synonyms', 'identical_modal']
    languages = args.languages
    seeds = args.seeds
    # batch_size = args.batch_size
    batch_size = 16  # fixed to avoid OOM error
    sample_size = args.sample_size
    version = args.version
    answer_type_ABC = args.answer_type_ABC

    # specify here which prompt structure you want to import
    module_name = f"prompts.templates.{args.prompt_structure_name}"
    module = __import__(module_name, fromlist=["prompt_templates"])
    prompt_templates = getattr(module, "prompt_templates")

    print(
        'experiment: first softmax over vocab other seeds, ABC maybe -> pickle {version}, ({models}; {tasks}; {prompt_types}; {languages})')
    print('****Start Time:', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    start_time = datetime.now()

    for seed in seeds:
        for lang in languages:
            file_path = f'./ATCS_group3/saved_outputs/logits_dict_seed_{seed}_lang_{lang}_v{version}.pickle'

            pipeline(seed, lang, models, tasks, prompt_types, prompt_templates,
                     batch_size, sample_size, answer_type_ABC, file_path=file_path)

    end_time = datetime.now()
    duration = end_time - start_time
    print('****End Time:', end_time, f'Duration: {duration}')
