from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
import torch
from dataloader_file import get_dataloader
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# Reduce the batch size to 16
batch_size = 16
# Load the dataset
dataset = load_dataset('benjaminvdb/DBRD', split='train')
# dataset = load_dataset('sst', split='train')
# Reduce the max_length to 128
dataloader = get_dataloader(
    dataset, tokenizer, batch_size=batch_size)


# dataset = load_dataset('sst', split='train')
# # The model you are using, XLM-RoBERTa, expects a fixed batch size of 4096.This means that the batch size used to train the model must also be 4096.
# dataloader = get_dataloader(dataset, tokenizer, batch_size=4096)

# define the possible answers
possible_answers = ['positive', 'negative']
possible_answers_ids = [tokenizer.encode(
    answer) for answer in possible_answers]

# define the total and correct predictions counter
total_preds = 0
correct_preds = 0

# prepare input
for batch in dataloader:
    # print(batch)
    input_ids = batch['input_ids']  # .to(device)
    print(len(input_ids))
    attention_mask = batch['attention_mask']  # .to(device)
    labels = batch['label']  # .to(device)

    # forward pass
    output = model(input_ids=input_ids,
                   attention_mask=attention_mask, labels=labels)
    print(output)

    logits = output.logits[:, -1, :]

    # Get the probabilities of all tokens in the vocabulary
    probabilities = logits.softmax(dim=-1)

    # calculate the probabilities of possible answers
    answers_probs = []
    for idx, answer_id in enumerate(possible_answers_ids):
        probs = probabilities[0, answer_id[-1]].item()
        answers_probs.append(probs)

    # determine the predicted answer
    pred_answer = possible_answers[answers_probs.index(max(answers_probs))]

    # get the true answer
    true_answer = 'positive' if labels.item() else 'negative'

    # check if the predicted answer is correct
    if pred_answer == true_answer:
        correct_preds += 1
    total_preds += 1

# calculate the accuracy
accuracy = correct_preds / total_preds
print(f"Accuracy: {accuracy:.4f}")
