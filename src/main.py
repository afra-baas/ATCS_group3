import argparse
# from config import data, model, task
from models.model import Model
from data.MARC.dataloader import MARCDataLoader
from data.NLI.dataloader import NLIDataLoader
from eval import evaluate


def pipeline(args):
    LM_model = args.LM_model
    task = args.task

    # Initilize model
    LM = Model(LM_model)
    batch_size = 8
    sample_size = 20
    if task == 'SA':
        train_dataloader = MARCDataLoader(sample_size, batch_size)
    elif task == 'NLI':
        train_dataloader = NLIDataLoader(sample_size, batch_size)
    else:
        print('This task evaluation is not implemented')

    answers_probs_all = []
    pred_answer_all = []
    mapped_labels_all = []
    i = 0

    possible_answers = train_dataloader.possible_answers
    for batch in train_dataloader:
        print(
            f'Batch: {i} , batch size: {batch_size}, sample_size: {sample_size}')
        prompts, mapped_labels = batch

        # Classification
        answers_probs_batch, pred_answer_batch = LM(
            prompts, possible_answers)
        print(f'pred_answer {pred_answer_batch} , label: {mapped_labels}')

        answers_probs_all.extend(answers_probs_batch)
        pred_answer_all.extend(pred_answer_batch)
        mapped_labels_all.extend(mapped_labels)

        i += 1

    # Evaluation
    acc = evaluate(pred_answer_all, mapped_labels_all)
    print('acc: ', acc)
    return acc


if __name__ == "__main__":
    # Args
    #   prompt_instructions: list of strings
    #   prompt_querry: list of strings
    #   label_map: dict
    #   LM_model: string
    #   task: string
    #
    # DEFAULT_MODEL = model["DEFAULT_MODEL"]
    # DEFAULT_TASK = task["DEFAULT_TASK"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--LM_model", type=str,
                        default='bloom')
    parser.add_argument("--task", type=str, default='SA')
    args = parser.parse_args()
    pipeline(args)
