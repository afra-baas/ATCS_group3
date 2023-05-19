import matplotlib.pyplot as plt
import pickle
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class AccuracyVisualizer:
    def __init__(self, data, models, languages, prompts):
        self.data = data
        self.models = models
        self.languages = languages
        self.prompts = prompts

    def visualize(self, file=f'./ATCS_group3/saved_outputs/Accuracies_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'):
        bar_width = 0.2
        opacity = 0.8
        x_pos = np.arange(len(self.models))

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, language in enumerate(self.languages):
            for j, prompt in enumerate(self.prompts):
                accuracies = self.data[(language, prompt)]
                offsets = x_pos + \
                    (i * bar_width * len(self.prompts)) + (j * bar_width)
                ax.bar(offsets, accuracies, bar_width, alpha=opacity,
                       label=f'{language} - {prompt} prompt')

        ax.set_xticks(x_pos + (bar_width * len(self.prompts) / 2))
        ax.set_xticklabels(self.models, rotation=45)
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison')
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
    print('num_prompt_vars', num_prompt_vars)

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
    print('num_prompt_vars', num_prompt_vars)

    mean_prompt_type_acc = 0
    for i in range(num_prompt_vars):
        prompt_dict = reduce(
            lambda d, k: d[k], path + [f'prompt_id_{i}'], data)
        acc = prompt_dict['acc']
        mean_prompt_type_acc += acc

    mean_prompt_type_acc = mean_prompt_type_acc / num_prompt_vars

    # To Do: significance measure
    return mean_prompt_type_acc


def save_dict_to_txt(dictionary, file_path):
    with open(file_path, 'w+') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")


def get_acc_plot(languages, models, prompt_types, task, seed):
    with open('./ATCS_group3/saved_outputs/logits_dict.pickle', 'rb') as f:
        data = pickle.load(f)

    # save_dict_to_txt(data, './ATCS_group3/saved_outputs/logits_dict.txt')

    # sample_size
    batch = data['42']['en']['bloom']['SA']['active']['prompt_id_1']
    print('batch', batch)

    lang_map = {'English': 'en', 'German': 'de'}

    plot_data = {}
    for lang in languages:
        for LM_model in models:
            for prompt_type in prompt_types:
                key = (lang, prompt_type)
                if key in plot_data:
                    # Append data to existing key
                    plot_data[key].extend(get_acc_from_logits(
                        data, model=LM_model, task=task, lang=lang_map[lang], seed=seed, prompt_type=prompt_type))
                else:
                    # Create new key and assign data
                    plot_data[key] = get_acc_from_logits(
                        data, model=LM_model, task=task, lang=lang_map[lang], seed=seed, prompt_type=prompt_type)

    visualizer = AccuracyVisualizer(plot_data, models, languages, prompt_types)
    visualizer.visualize()

##
    plot_data = {}
    for lang in languages:
        for LM_model in models:
            for prompt_type in prompt_types:
                key = (lang, prompt_type)
                if key in plot_data:
                    # Append data to existing key
                    plot_data[key].extend(get_acc_from_logits2(
                        data, model=LM_model, task=task, lang=lang_map[lang], seed=seed, prompt_type=prompt_type))
                else:
                    # Create new key and assign data
                    plot_data[key] = get_acc_from_logits2(
                        data, model=LM_model, task=task, lang=lang_map[lang], seed=seed, prompt_type=prompt_type)

    visualizer = AccuracyVisualizer(plot_data, models, languages, prompt_types)
    visualizer.visualize(
        file=f'./ATCS_group3/saved_outputs/Accuracies2_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')


############################### GET BOX PLOT INFO ########################################


class BoxPlotVisualizer:
    def __init__(self, data):
        self.data = data

    def visualize(self, box_plot_names, num_plots, file=f'./ATCS_group3/saved_outputs/Box_plots_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'):
        fig, ax = plt.subplots()
        ax.boxplot(self.data)
        ax.set_xlabel('Tasks and Sentence Types')
        ax.set_ylabel('Difference')
        ax.set_title(
            'Comparison of Differences between yes and no probability')
        ax.set_xticks(num_plots)
        ax.set_xticklabels(box_plot_names)

        plt.tight_layout()
        plt.savefig(file)
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


def get_box_plot(boxplots, box_plot_names):
    # this function expects a list of dictionaries
    with open('./ATCS_group3/saved_outputs/logits_dict.pickle', 'rb') as f:
        data = pickle.load(f)

    plot_data = []
    for boxplot in boxplots:
        plot_type = boxplot['type']
        print(boxplot)
        if plot_type == 'conditioned':
            plot_data.append(conditioned_all_sentence_boxplot(data, model=boxplot['model'], task=boxplot['task'], lang=boxplot[
                'lang'], seed=boxplot['seed'], prompt_type=boxplot['prompt_type'], condition=boxplot['condition']))
        elif plot_type == 'all':
            plot_data.append(all_sentence_boxplot(data, model=boxplot['model'], task=boxplot['task'],
                                                  lang=boxplot['lang'], seed=boxplot['seed'], prompt_type=boxplot['prompt_type']))
        elif plot_type == 'one':
            plot_data.append(one_sentence_boxplot(data, sen_id=boxplot['sen_id'], model=boxplot['model'],
                                                  task=boxplot['task'], lang=boxplot['lang'], seed=boxplot['seed'], prompt_type=boxplot['prompt_type']))
        else:
            print(f'There is no boxplot definition for {plot_type}')

    num_plots = [i+1 for i in range(len(box_plot_names))]
    visualizer = BoxPlotVisualizer(plot_data)
    visualizer.visualize(box_plot_names, num_plots)
