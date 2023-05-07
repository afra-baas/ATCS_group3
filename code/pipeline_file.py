import argparse
from dataloaders import create_dataloader, create_dataloader_nli
from main_model import Classifier
from datasets import load_dataset


def label_mapping(labels, map):
    """
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
        promt = "We will give you a set of instructions an input sentence and a querry. "
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
        if predictions[i] == targets[i]:
            correct += 1

    accuracy = correct / total
    return accuracy


def pipeline(Datasetpath, prompt_instructions, prompt_querry, label_map, LM_model, task):

    # Initilize model
    model = Classifier(LM_model)

    # ToDo Dataloader
    # sentences, labels = Dataloader(datasetpath)

    if task == 'SA':
        train_dataloader = create_dataloader(
            32, "French", "train", model.tokenizer)
    elif task == 'NLI':
        # Load the XNLI dataset for all_languages
        dataset = load_dataset("xnli", 'fr')  # "all_languages")
        train_dataloader = create_dataloader_nli(
            dataset['train'], model.tokenizer)
    # elif task == 'SA2':
        # dataset = load_dataset('benjaminvdb/DBRD')
        # dataset = load_dataset('sst')
        # dataloader = get_dataloader(dataset, tokenizer, batch_size=4096)

    else:
        print('This task evaluation is not implemented')

    # for batch in train_dataloader:
    #     sentences, labels = batch
    #     print('sentences, labels: ', sentences, labels)
    #     break

    answers_probs_all = []
    pred_answer_all = []
    mapped_labels_all = []
    for batch in train_dataloader:
        sentences, labels = batch

        # Generate promts
        prompts = prompt_generator(
            prompt_instructions, prompt_querry, sentences)

        # map labels
        mapped_labels = label_mapping(labels, label_map)

        answers_probs_batch = []
        pred_answer_batch = []
        for prompt in prompts:
            # ToDo classificaton
            answers_probs, pred_answer = model(prompt, list(label_map.keys()))
            answers_probs_batch.append(answers_probs)
            pred_answer_batch.append(pred_answer)

        answers_probs_all.extend(answers_probs_batch)
        pred_answer_all.extend(pred_answer_batch)
        mapped_labels_all.extend(mapped_labels)

    # ToDo Evaluation:
    acc = evaluate(pred_answer_all, mapped_labels_all)
    print('acc', acc)

    return acc


if __name__ == "__main__":

    Datasetpath = ''
    LM_model = 'xlm-roberta-base'
    task = 'SA'

    if task == 'SA':
        # prompt_instructions = [
        #     'Can you please tell me the sentiment of this review', ' Give me the sentiment of this review now']
        # prompt_querry = ['is it positive or nagative?',
        #                  ' positive or negative?']

        prompt_instructions = 'Can you please tell me the sentiment of this review'
        prompt_querry = 'is it positive or nagative?'

        label_map = {'5': ' positive', '4': ' positive', '3': ' positive',
                     '2': ' negative', '1': ' negative',  '0': ' negative'}
    elif task == 'NLI':
        prompt_instructions = ['', ' ']
        prompt_querry = [' ', ' ']
        label_map = {'': ' ', ' ': ' '}
    else:
        prompt_instructions = [' ']
        prompt_querry = [' ']
        label_map = {'': ' '}

    acc = pipeline(Datasetpath, prompt_instructions,
                   prompt_querry, label_map, LM_model, task)

    # parser = argparse.ArgumentParser(description='Description of your program')

    # parser.add_argument('--datasetpath', type=str, help='Path to the dataset')
    # parser.add_argument('--lm_model', type=str,
    #                     help='Path to the language model')
    # args = parser.parse_args()

    # acc = pipeline(args.Datasetpath, prompt_instructions,
    #                prompt_querry, label_map, args.LM_model)

    # parser.add_argument('--datasetpath', type=str, help='Path to the dataset')
    # parser.add_argument('--prompt_instructions', type=str, help='Instructions for the prompt')
    # parser.add_argument('--prompt_querry', type=str, help='Query for the prompt')
    # parser.add_argument('--label_map', type=str, help='Path to the label map')
    # parser.add_argument('--lm_model', type=str, help='Path to the language model')

    # args = parser.parse_args()

    # acc= pipeline(args.Datasetpath, args.prompt_instructions, args.prompt_querry, args.label_map, args.LM_model)
