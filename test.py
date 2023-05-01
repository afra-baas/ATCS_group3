from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from LMmodel import CL_model
# from transformers import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

test = ["Review: I really like this movie. Is this review positive? Yes or No? Answer:"]
possible_answers = ["yes", "no"]
# model = CL_model("bigscience/bloom-1b1")
# print(model(test, possible_answers))
model = CL_model("bigscience/bloom-560m")
print(model(test, possible_answers))
# inputs = tokenizer(prompt, return_tensors="pt")
# possible_answers = ["yes", "no"]
# model = bloom_model()
# output = model(prompt, possible_answers)
# print(output)