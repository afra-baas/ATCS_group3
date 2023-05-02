# pip install -q transformers
from transformers import pipeline


# checkpoint = "MBZUAI/LaMini-T5-61M"
# model = pipeline('text2text-generation', model=checkpoint)

# # input_prompt = 'Please let me know your thoughts on the given place and why you think it deserves to be visited: \n"Barcelona, Spain"'
# input_prompt = 'what is the sentiment of this sentence: "i like trains.""'
# generated_text = model(input_prompt, max_length=512, do_sample=True)[
#     0]['generated_text']

# print(generated_text)


checkpoint = "MBZUAI/LaMini-T5-61M"
model = pipeline('text-generation', model=checkpoint)

instruction = 'Please let me know your thoughts on the given place and why you think it deserves to be visited: \n"Barcelona, Spain"'
input_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

generated_text = model(input_prompt, max_length=512, do_sample=True)[
    0]['generated_text']

print("Response", generated_text)
