import matplotlib.pyplot as plt
import pickle
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
import io
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def open_data_pickle(filename='logits_dict_seed_42_lang_de_v1.pickle'):
    with open(filename, 'rb') as f:
        # data = CPU_Unpickler(f).load() # if you run locally without gpu
        data = pickle.load(f)
    return data


class AccuracyVisualizer:
    def __init__(self, data, models, languages, prompts, task):
        self.data = data
        self.models = models
        lang_map = {'en': 'English', 'de': 'German'}
        self.languages = [lang_map[lang] for lang in languages]
        self.prompts = prompts
        self.task = task

    def visualize(self, file='./ATCS_group3/saved_outputs/Accuracies_v1.png'):
        space_between_bars = 0.15  # Adjust the value as needed
        opacity = 0.8
        x_pos = np.arange(len(self.models))
        model_offset = space_between_bars * \
            (len(self.languages) * len(self.prompts) - 1)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, language in enumerate(self.languages):
            for j, prompt in enumerate(self.prompts):
                accuracies = self.data[(language, prompt)]
                offsets = x_pos + \
                    (i * model_offset) + (j * space_between_bars)
                ax.bar(offsets, accuracies, space_between_bars, alpha=opacity,
                       label=f'{language} - {prompt} prompt')

        ax.set_xticks(x_pos + (model_offset / 2))
        ax.set_xticklabels(self.models, rotation=45)
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy Comparison for Task {self.task}')

        # Move the legend outside the plot
        # ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        ax.legend()

        plt.tight_layout()
        plt.savefig(file)
        plt.close()  # Close the figure to free up memory


def get_acc_from_logits(data, model='bloom', task='SA', lang='en', seed='42', prompt_type='active'):
    """
    Computes accuracy given predicted and target labels
    input:
        predictions: a list of predicted labels
        targets: a list of true labels
    output:
        accuracy: float value of the classification accuracy
    """
    path = [seed, lang, model, task, prompt_type]
    all_prompt_vars = reduce(lambda d, k: d[k], path, data)
    num_prompt_vars = len(all_prompt_vars.keys())

    targets = []
    predictions = []
    for i in range(num_prompt_vars):
        num_sentences = len(
            reduce(lambda d, k: d[k], path + [f'prompt_id_{i}'], data))-1
        for sen_id in range(num_sentences):
            sentence = reduce(
                lambda d, k: d[k], path + [f'prompt_id_{i}', sen_id], data)
            pred = sentence['pred']
            true = sentence['true']
            targets.append(true)
            predictions.append(pred)

    correct = sum(pred == true for pred, true in zip(predictions, targets))
    total = len(predictions)
    accuracy = correct / total

    # To Do: significance measure
    # Save results somewhere

    # returns mean_prompt_type_acc
    return accuracy


def get_acc_from_logits2(data, model='bloom', task='SA', lang='en', seed='42', prompt_type='active'):
    """
    Computes accuracy given predicted and target labels
    input:
        predictions: a list of predicted labels
        targets: a list of true labels
    output:
        accuracy: float value of the classification accuracy
    """
    path = [seed, lang, model, task, prompt_type]
    all_prompt_vars = reduce(lambda d, k: d[k], path, data)
    num_prompt_vars = len(all_prompt_vars.keys())

    mean_prompt_type_acc = 0
    for i in range(num_prompt_vars):
        prompt_dict = reduce(
            lambda d, k: d[k], path + [f'prompt_id_{i}'], data)
        acc = prompt_dict['acc']
        mean_prompt_type_acc += acc

    mean_prompt_type_acc = mean_prompt_type_acc / num_prompt_vars

    # To Do: significance measure
    return mean_prompt_type_acc


def get_acc_from_logits3(data, model='bloom', task='SA', lang='en', seed='42', prompt_type='active'):
    """
    Computes accuracy per prompt type and per prompt id within
    input:
        predictions: a list of predicted labels
        targets: a list of true labels
    output:
        accuracy: float value of the classification accuracy
    """
    path = [seed, lang, model, task, prompt_type]
    all_prompt_vars = reduce(lambda d, k: d[k], path, data)
    num_prompt_vars = len(all_prompt_vars.keys())

    list_prompt_type_acc = []
    for i in range(num_prompt_vars):
        prompt_dict = reduce(
            lambda d, k: d[k], path + [f'prompt_id_{i}'], data)
        acc = prompt_dict['acc']
        list_prompt_type_acc.append(acc)

    return list_prompt_type_acc


def get_acc_plot(languages, models, prompt_types, task, seed, version, file_path='./ATCS_group3/saved_outputs/logits_dict.pickle', fn=get_acc_from_logits):
    data = open_data_pickle(filename=file_path)
    lang_map = {'en': 'English', 'de': 'German'}

    plot_data = {}
    for lang in languages:
        for LM_model in models:
            for prompt_type in prompt_types:
                key = (lang_map[lang], prompt_type)
                if key in plot_data:
                    print('key', key)
                    if fn == get_acc_from_logits3:
                        plot_data[key].extend(fn(
                            data, model=LM_model, task=task, lang=lang, seed=seed, prompt_type=prompt_type))
                    else:
                        # Append data to existing key
                        plot_data[key].append(fn(
                            data, model=LM_model, task=task, lang=lang, seed=seed, prompt_type=prompt_type))
                else:
                    if fn == get_acc_from_logits3:
                        plot_data[key] = fn(
                            data, model=LM_model, task=task, lang=lang, seed=seed, prompt_type=prompt_type)
                    else:
                        # Create new key and assign data
                        plot_data[key] = [fn(
                            data, model=LM_model, task=task, lang=lang, seed=seed, prompt_type=prompt_type)]

    visualizer = AccuracyVisualizer(
        plot_data, models, languages, prompt_types, task)
    visualizer.visualize(file=f'Accuracies_v{version}_{fn}.png')


############################### GET BOX PLOT INFO ########################################

class BoxPlotVisualizer:
    def __init__(self, data):
        self.data = data

    def visualize(self, box_plot_names, num_plots, file=f'Box_plots_v1.png'):
        fig, ax = plt.subplots()
        ax.boxplot(self.data)
        ax.set_xlabel('Tasks and Sentence Types')
        ax.set_ylabel('Difference')
        ax.set_title(
            'Comparison of Differences between yes and no probability')
        ax.set_xticks(num_plots)
        ax.set_xticklabels(box_plot_names, rotation=45)

        plt.tight_layout()
        plt.show()
        plt.close()  # Close the figure to free up memory


def one_sentence_boxplot(data, sen_id, model='bloom', task='SA', lang='en', seed='42', prompt_type='active'):
    path = [seed, lang, model, task, prompt_type]
    all_prompt_vars = reduce(lambda d, k: d[k], path, data)
    num_prompt_vars = len(all_prompt_vars.keys())

    var_for_each_prompt = []
    for i in range(num_prompt_vars):
        sentence = reduce(
            lambda d, k: d[k], path + [f'prompt_id_{i}', sen_id], data)
        var = sentence['diff']
        var_for_each_prompt.append(var)
    print('len(var_for_each_prompt)', len(var_for_each_prompt))
    return var_for_each_prompt


def one_sentence_per_type_boxplot(data, sen_id, model='bloom', task='SA', lang='en', seed='42'):
    path = [seed, lang, model, task]
    all_prompt_types = reduce(lambda d, k: d[k], path, data)
    num_prompt_types = len(all_prompt_types.keys())

    var_for_each_prompt = []
    for i in range(num_prompt_types):
        prompt_type = num_prompt_types[i]
        num_prompt_vars = len(
            reduce(lambda d, k: d[k], path + [prompt_type], data))
        for i in range(num_prompt_vars):
            sentence = reduce(
                lambda d, k: d[k], path + [f'prompt_id_{i}', sen_id], data)
            var = sentence['diff']
            var_for_each_prompt.append(var)

    print('len(var_for_each_prompt)', len(var_for_each_prompt))
    return var_for_each_prompt


def all_sentence_boxplot(data, model='bloom', task='SA', lang='en', seed='42', prompt_type='active'):
    path = [seed, lang, model, task, prompt_type]
    all_prompt_vars = reduce(lambda d, k: d[k], path, data)
    num_prompt_vars = len(all_prompt_vars.keys())

    var_for_each_prompt = []
    for i in range(num_prompt_vars):
        num_sentences = len(
            reduce(lambda d, k: d[k], path + [f'prompt_id_{i}'], data))-1
        for sen_id in range(num_sentences):
            sentence = reduce(
                lambda d, k: d[k], path + [f'prompt_id_{i}', sen_id], data)
            var = sentence['diff']
            var_for_each_prompt.append(var)
    print('len(var_for_each_prompt)', len(var_for_each_prompt))
    return var_for_each_prompt


def conditioned_all_sentence_boxplot(data, model='bloom', task='SA', lang='en', seed='42', prompt_type='active', condition='yes'):
    path = [seed, lang, model, task, prompt_type]
    all_prompt_vars = reduce(lambda d, k: d[k], path, data)
    num_prompt_vars = len(all_prompt_vars.keys())

    var_for_each_prompt = []
    for i in range(num_prompt_vars):
        num_sentences = len(
            reduce(lambda d, k: d[k], path + [f'prompt_id_{i}'], data))-1
        for sen_id in range(num_sentences):
            sentence = reduce(
                lambda d, k: d[k], path + [f'prompt_id_{i}', sen_id], data)
            if sentence['pred'] == condition:
                var = sentence['diff']
                var_for_each_prompt.append(var)
    print('len(var_for_each_prompt)', len(var_for_each_prompt))
    return var_for_each_prompt


def get_box_plot(boxplots, box_plot_names, version, file_path='logits_dict.pickle'):
    # this function expects boxplots to be a list of dictionaries
    data = open_data_pickle(filename=file_path)

    plot_data = []
    for boxplot in boxplots:
        plot_type = boxplot['type']
        # print(boxplot)
        if plot_type == 'conditioned':
            plot_data.append(conditioned_all_sentence_boxplot(data, model=boxplot['model'], task=boxplot['task'], lang=boxplot[
                'lang'], seed=boxplot['seed'], prompt_type=boxplot['prompt_type'], condition=boxplot['condition']))
        elif plot_type == 'all':
            plot_data.append(all_sentence_boxplot(data, model=boxplot['model'], task=boxplot['task'],
                                                  lang=boxplot['lang'], seed=boxplot['seed'], prompt_type=boxplot['prompt_type']))
        elif plot_type == 'one':
            plot_data.append(one_sentence_boxplot(data, sen_id=boxplot['sen_id'], model=boxplot['model'],
                                                  task=boxplot['task'], lang=boxplot['lang'], seed=boxplot['seed'], prompt_type=boxplot['prompt_type']))
        elif plot_type == 'one_per':
            plot_data.append(one_sentence_per_type_boxplot(data, sen_id=boxplot['sen_id'], model=boxplot['model'],
                                                           task=boxplot['task'], lang=boxplot['lang'], seed=boxplot['seed'], prompt_type=boxplot['prompt_type']))
        else:
            print(f'There is no boxplot definition for {plot_type}')

    num_plots = [i+1 for i in range(len(box_plot_names))]
    visualizer = BoxPlotVisualizer(plot_data)
    visualizer.visualize(box_plot_names, num_plots,
                         file=f'Box_plots_v{version}.png')


# friedmanchisquare is for testing significance between prompts in same prompt type
def get_friedman_test(file_path='./ATCS_group3/saved_outputs/logits_dict.pickle'):
    data = open_data_pickle(filename=file_path)
    all_acc_for_prompt_type = one_sentence_boxplot(
        data, 1, model='bloom', task='SA', lang='de', seed='42', prompt_type='active')
    acc_list_of_lists = [[acc] for acc in all_acc_for_prompt_type]
    print(acc_list_of_lists)
    score, p_value = friedmanchisquare(*acc_list_of_lists)

    print("Friedman chi-square score:", score)
    print("p-value:", p_value)

    return score, p_value


def get_acc_per_model(lang, LM_model, prompt_type, task, seed, file_path='./ATCS_group3/saved_outputs/logits_dict.pickle', fn=get_acc_from_logits):
    data = open_data_pickle(filename=file_path)
    acc = fn(data, model=LM_model, task=task, lang=lang,
             seed=seed, prompt_type=prompt_type)
    print('acc', acc)

    return acc


def get_wilcoxon_test(file_path='./ATCS_group3/saved_outputs/logits_dict.pickle'):
    acc_active = get_acc_per_model('de', 'bloom', 'active',
                                   'SA', '42', file_path=file_path, fn=get_acc_from_logits)
    acc_passive = get_acc_per_model('de', 'bloom', 'passive',
                                    'SA', '42', file_path=file_path, fn=get_acc_from_logits)

    statistic, p_value = wilcoxon([acc_active], [acc_passive])

    print("Test Statistic:", statistic)
    print("p-value:", p_value)
    return statistic, p_value


if __name__ == "__main__":
    models = ['bloom', 'bloomz', 'flan', 'llama']  # , 'alpaca']
    # models = ['bloom']
    tasks = ['SA', 'NLI']
    prompt_types = ['active', 'passive', 'auxiliary', 'modal', 'rare_synonyms']
    # prompt_types = ['active', 'passive']
    # languages = ['en', 'de']
    languages = ['de']
    # seeds = ['42', '33', '50']
    seeds = ['42']

    # # NOTE: dont give a list here
    task = 'SA'
    seed = '42'
    lang = 'en'
    version = 1
    file_path = f'./ATCS_group3/saved_outputs/logits_dict_seed_{seed}_lang_{lang}_v{version}.pickle'

    ######## ACC plots ##########
    get_acc_plot(languages, models, prompt_types,
                 task, seed, version, file_path=file_path, fn=get_acc_from_logits)

    ####### Box Plots #########

    # list of dictionaries
    boxplots = [{'type': 'conditioned', 'model': 'bloom', 'task': task, 'lang': lang,
                 'seed': '42', 'prompt_type': 'active', 'condition': 'yes'},

                {'type': 'conditioned', 'model': 'bloom', 'task': task, 'lang': lang,
                    'seed': '42', 'prompt_type': 'active', 'condition': 'no'},

                {'type': 'all', 'model': 'bloom', 'task': task, 'lang': lang,
                    'seed': '42', 'prompt_type': 'active'},

                {'type': 'one', 'sen_id': 1, 'model': 'bloom', 'task': task, 'lang': lang,
                    'seed': '42', 'prompt_type': 'active'},

                {'type': 'one_per', 'sen_id': 1, 'model': 'bloom', 'task': task, 'lang': lang,
                    'seed': '42', 'prompt_type': 'active'}]

    box_plot_names = ['Diff conditioned yes', 'Diff conditioned no',
                      'Diff all sentences', 'Diff one sentence', 'Diff one sentence per prompt']

    get_box_plot(boxplots, box_plot_names, version, file_path=file_path)

    ##### significance tests #####

    score, p_value = get_friedman_test(file_path=file_path)
    statistic, p_value = get_wilcoxon_test(file_path=file_path)
