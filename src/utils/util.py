import argparse
from dataloaders import create_dataloader, create_dataloader_nli
from src.models.model import Model
from datasets import load_dataset
from datetime import datetime
from typing import Dict
from config import data, model
# from ATCS_group3.src.config import data, model


def label_mapping(labels: list, map: Dict[str, str]):
    """
    :param labels: list of true labels
    :param map: a dict where each key represents a true label and each value is the mapping value
    # TODO: check actual value of labels
    :return: a list of mapped labels.
    Maps labels from data loader to the desired labels.
    input:
        labels: list of true labels
        map: a dict where each key represents a true label and each value is the mapping value
    output:
        output: a list of mapped labels.
    """
    output = []
    for label in labels:
        label = str(label.item())
        if label not in map:
            print(f"couldn't find {label} in the mapping")
            output.append(None)
        else:
            output.append(map[label])
    return output


def prompt_generator(prompt_instructions, prompt_querry, sentences):
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
    for sentence in sentences:
        promt = (
            "We will give you a set of instructions an input sentence and a querry. "
        )
        promt += "You should answer the querry based on the input sentence accord to the instructions provided. \n"
        promt += f"instructions: {prompt_instructions} \ninput sentence: {sentence} \nquerry: {prompt_querry} \nanswer: "
        output.append(promt)
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
        # print(predictions[i], targets[i], type(predictions[i]),
        #       type(targets[i]), predictions[i] == targets[i])
        if predictions[i] == targets[i]:
            correct += 1

    accuracy = correct / total
    return accuracy


def pipeline(prompt_instructions, prompt_querry, label_map, LM_model, task):

    # Initilize model
    model = Model(LM_model)

    if task == "SA":
        train_dataloader = create_dataloader(32, "French", "train", model.tokenizer)
    elif task == "NLI":
        # Load the XNLI dataset for all_languages
        dataset = load_dataset("xnli", "fr")  # "all_languages")
        train_dataloader = create_dataloader_nli(dataset["train"], model.tokenizer)
    # elif task == 'SA2':
    # dataset = load_dataset('benjaminvdb/DBRD')
    # dataset = load_dataset('sst')
    # dataloader = get_dataloader(dataset, tokenizer, batch_size=4096)
    else:
        print("This task evaluation is not implemented")

    answers_probs_all = []
    pred_answer_all = []
    mapped_labels_all = []
    i = 0
    for batch in train_dataloader:
        sentences, labels = batch

        start_time = datetime.now()
        # Generate promts
        prompts = prompt_generator(prompt_instructions, prompt_querry, sentences)
        print(f"Batch number: {i} , batch size : {len(prompts)}")

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Time taken to execute prompt gen: {duration}")

        start_time = datetime.now()
        # map labels
        mapped_labels = label_mapping(labels, label_map)

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Time taken to execute mapping: {duration}")

        # Classification
        start_time = datetime.now()
        answers_probs_batch, pred_answer_batch = model(
            prompts, ["positive", "negative"]
        )

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Time taken to execute classification (32): {duration}")
        print("pred_answer ", pred_answer_batch)
        print("mapped_labels ", mapped_labels)

        answers_probs_all.extend(answers_probs_batch)
        pred_answer_all.extend(pred_answer_batch)
        mapped_labels_all.extend(mapped_labels)

        acc = evaluate(pred_answer_batch, mapped_labels)
        print("Batch acc: ", acc)
        print()

        i = +1
        if i > 2:
            break

    start_time = datetime.now()
    # Evaluation:
    acc = evaluate(pred_answer_all, mapped_labels_all)
    print("acc: ", acc)
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Time taken to execute eval: {duration}")

    return acc


if __name__ == "__main__":

    LM_model = "xlm-roberta-base"
    # LM_model = 'bigscience/bloom-560m'
    task = "SA"

    if task == "SA":
        prompt_instructions = "Can you please tell me the sentiment of this review"
        prompt_querry = "is it positive or nagative?"

        label_map = {
            "5": "positive",
            "4": "positive",
            "3": "positive",
            "2": "negative",
            "1": "negative",
            "0": "negative",
        }
    elif task == "NLI":
        prompt_instructions = ["", " "]
        prompt_querry = [" ", " "]
        label_map = {"": " ", " ": " "}
    else:
        prompt_instructions = [" "]
        prompt_querry = [" "]
        label_map = {"": " "}

    acc = pipeline(prompt_instructions, prompt_querry, label_map, LM_model, task)

    # parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('--datasetpath', type=str, help='Path to the dataset')
    # parser.add_argument('--lm_model', type=str,
    #                     help='Path to the language model')
    # args = parser.parse_args()

    # acc = pipeline(args.Datasetpath, prompt_instructions,
    #                prompt_querry, label_map, args.LM_model)
