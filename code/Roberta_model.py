from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from dataloader_file import get_dataloader
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")


# define the possible answers
possible_answers = ['positive', 'negative']
possible_answers_ids = [tokenizer.encode(
    answer) for answer in possible_answers]

prompt = 'is this review positive or negative: '
sample = 'I really liked this movie'
text = f"{prompt} {sample}"
inputs = tokenizer(text, return_tensors="pt")

# forward pass
output = model(**inputs, labels=inputs["input_ids"])
print(output)

logits = output.logits[:, -1, :]

# Get the probabilities of all tokens in the vocabulary
probabilities = logits.softmax(dim=-1)

# calculate the probabilities of possible answers
answers_probs = []
for idx, answer_id in enumerate(possible_answers_ids):
    probs = probabilities[0, answer_id[-1]].item()
    answers_probs.append(probs)
print('answers_probs', answers_probs)

# determine the predicted answer
pred_answer = possible_answers[answers_probs.index(max(answers_probs))]
print('pred_answer', pred_answer)

# get the true answer
# true_answer = 'positive' if labels.item() else 'negative'
