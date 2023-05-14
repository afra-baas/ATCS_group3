import argparse
from dataloaders import create_dataloader, create_dataloader_nli
from main_model import Classifier
from datasets import load_dataset
from datetime import datetime
import torch


def label_mapping(labels):
    """
    Maps labels from data loader to the desired labels.
    input:
        labels: list of true labels
        map: a dict where each key represents a true label and each value is the mapping value
    output:
        output: a list of mapped labels.
    """

    if task == 'SA':
        map = {'5': 'positive', '4': 'positive', '3': 'positive',
               '2': 'negative', '1': 'negative',  '0': 'negative'}
    elif task == 'NLI':
        map = {'0': 'entailment', '1': 'neutral',
               '2': 'contradiction'}

    output = []
    for label in labels:
        label = str(label.item())
        if label not in map:
            print(f"couldn't find {label} in the mapping")
            output.append(None)
        else:
            output.append(map[label])

    possible_answers = list(set(map.values()))
    return output, possible_answers


def prompt_generator(sentences):
    """
    Generates a promt for every sentence according to the instructions provided
    input:
        prompt_instructions: a str with the general prompt instructions.
        prompt_querry: a str with the question we want to ask the language model
        sentences: a list with all the input sentences
    output:
        output: a list with all sentences transformed to the desired prompt.
    """

    prompt_instructions = 'Can you please tell me the sentiment of this review'
    prompt_querry = 'is it positive or nagative?'

    output = []
    for sentence in sentences:
        prompt = "We will give you a set of instructions an input sentence and a querry. "
        prompt += "You should answer the querry based on the input sentence accord to the instructions provided. \n"
        prompt += f"instructions: {prompt_instructions} \ninput sentence: {sentence} \nquerry: {prompt_querry} \nanswer: "
        output.append(prompt)
    return output


# def nli_prompt_generator(sentences):
#     """
#     Generates a promt for every sentence according to the instructions provided
#     input:
#         prompt_instructions: a str with the general prompt instructions.
#         prompt_querry: a str with the question we want to ask the language model
#         sentences: a list with all the input sentences
#     output:
#         output: a list with all sentences transformed to the desired prompt.
#     """
#     output = []
#     premises, hypotheses = zip(*sentences)
#     for i in range(len(sentences)):
#         prompt = f"{premises[i]} \n Question: {hypotheses[i]} True, False, or Neither?"
#         output.append(prompt)
#     return output

def nli_prompt_generator(premises, hypotheses):
    """
    Generates a promt for every sentence according to the instructions provided
    input:
        prompt_instructions: a str with the general prompt instructions.
        prompt_querry: a str with the question we want to ask the language model
        sentences: a list with all the input sentences
    output:
        output: a list with all sentences transformed to the desired prompt.
    """
    output = []
    for i in range(len(premises)):
        prompt = f"{premises[i]} \n Question: {hypotheses[i]} True, False, or Neither?"
        output.append(prompt)
    return output


def evaluate(predictions, targets):
    """
    Computes accuracy given predicted and target labels
    input:
        predictions: a list of predicted labels
        targets: a list of true labels
    output:
        accuracy: float value of the classification accuracy
    """
    correct = 0
    total = len(predictions)

    for i in range(total):
        if predictions[i] == targets[i]:
            correct += 1

    accuracy = correct / total
    return accuracy


def pipeline(LM_model, task, prompt_gen):

    # Initilize model
    model = Classifier(LM_model)
    batch_size = 1
    sample_size = 100
    if task == 'SA':
        train_dataloader = create_dataloader(sample_size, batch_size)
    elif task == 'NLI':
        train_dataloader = create_dataloader_nli(sample_size, batch_size)
    else:
        print('This task evaluation is not implemented')

    answers_probs_all = []
    pred_answer_all = []
    mapped_labels_all = []
    i = 0
    for batch in train_dataloader:
        print(
            f'Batch: {i} , batch size: {batch_size}, sample_size: {sample_size}')
        start_time = datetime.now()
        if task == 'SA':
            sentences, labels = batch
            start_time = datetime.now()
            # Generate promts
            prompts = prompt_gen(sentences)
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"Time taken to execute pipeline function: {duration}")

        else:
            premise, hypotheses, labels = batch
            start_time = datetime.now()
            # Generate promts
            prompts = prompt_gen(premise, hypotheses)
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"Time taken to execute pipeline function: {duration}")

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Time taken to execute prompt gen: {duration}")

        start_time = datetime.now()
        # map labels
        mapped_labels, possible_answers = label_mapping(labels)
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Time taken to execute mapping: {duration}")

        # Classification
        start_time = datetime.now()
        answers_probs_batch, pred_answer_batch = model(
            prompts, possible_answers)

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Time taken to execute classification (32): {duration}")
        print('pred_answer ', pred_answer_batch)
        print('mapped_labels ', mapped_labels)

        answers_probs_all.extend(answers_probs_batch)
        pred_answer_all.extend(pred_answer_batch)
        mapped_labels_all.extend(mapped_labels)

        acc = evaluate(pred_answer_batch, mapped_labels)
        print('Batch acc: ', acc)
        print()
        i += 1

    start_time = datetime.now()
    # Evaluation
    acc = evaluate(pred_answer_all, mapped_labels_all)
    print('acc: ', acc)
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Time taken to execute eval: {duration}")

    return acc


if __name__ == "__main__":
    LM_model = 'bigscience/bloom-560m'
    # task = 'NLI'
    task = 'SA'
    print('task ', task)

    if task == "NLI":
        acc = pipeline(LM_model, task, nli_prompt_generator)
    else:
        acc = pipeline(LM_model, task, prompt_generator)
    # parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('--lm_model', type=str,help='Path to the language model')
    # args = parser.parse_args()
    # acc = pipeline(args.LM_model)
