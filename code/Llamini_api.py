# pip install -q transformers
from transformers import pipeline
import torch
from transformers import T5Tokenizer, AutoTokenizer, AutoModelForCausalLM

checkpoint = "MBZUAI/LaMini-T5-61M"
model = pipeline('text-generation', model=checkpoint)

# model = AutoModelForCausalLM.from_pretrained("MBZUAI/LaMini-T5-61M")


# tokenizer = T5Tokenizer.from_pretrained("MBZUAI/LaMini-T5-61M")
tokenizer = AutoTokenizer.from_pretrained('MBZUAI/LaMini-T5-61M')


# Set the model to evaluation mode
# model.eval()

instruction = 'Please let me know your thoughts on the given place and why you think it deserves to be visited: \n"Barcelona, Spain"'
input_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"


# Generate the output sequence
# with torch.no_grad():
# output = model.generate(input_ids)
generated_text = model(input_prompt, max_length=512, do_sample=True)[
    0]['generated_text']


print("----------------Response", generated_text)

logits = generated_text.logits[:, -1, :]
probabilities = logits.softmax(dim=-1)

# Get the probabilities of all tokens in the vocabulary
# probabilities = torch.nn.functional.softmax(generated_text[0], dim=-1)


token_probabilities = [(tokenizer.decode(i.item()), p.item())
                       for i, p in enumerate(probabilities[0])]
