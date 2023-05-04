from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
import torch
from dataloader_file import get_dataloader
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# define the batch size and gradient accumulation steps
batch_size = 32
gradient_accumulation_steps = 128 // batch_size


dataset = load_dataset('sst', split='train')
# The model you are using, XLM-RoBERTa, expects a fixed batch size of 4096.This means that the batch size used to train the model must also be 4096.
dataloader = get_dataloader(dataset, tokenizer, batch_size=4096)

# define the possible answers
possible_answers = ['positive', 'negative']
possible_answers_ids = [tokenizer.encode(
    answer) for answer in possible_answers]

# define the total and correct predictions counter
total_preds = 0
correct_preds = 0

# define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()


# train the model
for step, batch in enumerate(dataloader):
    # prepare input
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label']

    # forward pass
    output = model(input_ids=input_ids,
                   attention_mask=attention_mask, labels=labels)
    loss = output.loss / gradient_accumulation_steps

    # backward pass
    loss.backward()
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    # calculate the probabilities of possible answers
    logits = output.logits[:, -1, :]
    probabilities = logits.softmax(dim=-1)
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

    # print the loss and accuracy every 100 steps
    if (step + 1) % 100 == 0:
        accuracy = correct_preds / total_preds
        print(
            f"Step {step+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# print the final accuracy
accuracy = correct_preds / total_preds
print(f"Final Accuracy: {accuracy:.4f}")
