from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from Bloommodel import CL_bloom
from label_mapping import label_mapping
from prompt_generator import prompt_generator
# from transformers import GenerationConfig

# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

# test = ["Review: I really like this movie. Is this review positive? Yes or No? Answer:"]
# possible_answers = ["yes", "no"]
# # model = CL_model("bigscience/bloom-1b1")
# # print(model(test, possible_answers))
# model = CL_bloom("bigscience/bloom-560m")
# print(model(test, possible_answers))
# # inputs = tokenizer(prompt, return_tensors="pt")
# # possible_answers = ["yes", "no"]
# # model = bloom_model()
# # output = model(prompt, possible_answers)
# # print(output)

prompt_instruction = "I want you to give me the sentiment of the follow review:"
# prompt_instruction = ""
prompt_querry = "Is this review positive or negative?"
# prompt_querry = ""
sentences = ["This movie was amazing.", "This must be the worst movie I have every seen.", "Could have been better, could have been worse."]

output = prompt_generator(prompt_instruction, prompt_querry, sentences)
print(output)
possible_answers = ["positive", "negative"]
model = CL_bloom()
print(model(output, possible_answers))