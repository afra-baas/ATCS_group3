from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

class CL_model():
    def __init__(self, model_name = "bigscience/bloom-560m"):
        """
        Initializes model
        input:
            model_name: name of the model
        """
        #load model and model tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def __call__(self, prompt, possible_answers):
        """
        Generate probabilites for each promt and each possible answer.
        Input: 
            prompt: list of strings where each string is a seperate prompt
            possible_answers: list of strings where each string is an answer we want the logits from

        Output:
            answer_probs: tensor of shape (len(prompt), len(possible_answer)) where the values are the logits for each answer per prompt
        """
        #tokenize input and possible answers
        inputs = self.tokenizer(prompt, return_tensors="pt")
        possible_answers_ids = [self.tokenizer(answer) for answer in possible_answers]

        #generate outputs
        outputs = self.model(**inputs, labels = inputs["input_ids"])

        #get the logits of the last token
        logits = outputs.logits[:, -1]

        #loop over all possible answers for every promt and store the logits
        answers_probs = torch.zeros(len(prompt), len(possible_answers_ids))
        for idx, answer in enumerate(possible_answers_ids):
            id = answer["input_ids"]
            probs = logits[:, id][0]
            answers_probs[:, idx] = probs

        return answers_probs
