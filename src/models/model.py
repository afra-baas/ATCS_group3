from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import torch
from typing import List
from config import model

class Model():
    def __init__(self, model_name: str, device: str ="cpu"):
        """
        :param model_name: name of the model to use
        :param device: device to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            model_config = model["SUPPORTED_MODELS"][model_name]
        except KeyError:
            raise KeyError(
                f"Model {model_name} not supported. Please use one of the following models: {model['SUPPORTED_MODELS'].keys()}"
            )
        self.model_name = model_config["model_name"]
        print(f"Loading model {self.model_name}")
        self.model = model_config["model_constructor"](self.model_name)
        print(f"Model {self.model_name} loaded")
        print(f"Moving model to {device}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, prompt: List[str], possible_answers):
        """
        Generate probabilites for each promt and each possible answer.
        :param prompt: a list of strings with the prompts
        :param possible_answers: a list of strings with the possible answers
        :return: a tensor of shape (len(prompt), len(possible_answer))  where the values are the logits for each answer per prompt
        """
        # tokenize input and possible answers
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True).to(self.device)
        possible_answers_ids = [self.tokenizer(answer) for answer in possible_answers]

        # generate outputs
        outputs = self.model(**inputs, labels=inputs["input_ids"])

        # get the logits of the last token
        logits = outputs.logits[:, -1]
        # loop over all possible answers for every promt and store the logits
        answers_probs = torch.zeros(len(prompt), len(
            possible_answers_ids)).to(self.device)

        for idx, answer in enumerate(possible_answers_ids):
            id = answer["input_ids"]
            probs = logits[:, id]
            answers_probs[:, idx] = probs.T

        # pred_answer = possible_answers[answers_probs.index(max(answers_probs))]
        pred_answer_indices = answers_probs.argmax(dim=1)
        pred_answer = [possible_answers[i] for i in pred_answer_indices]
        print('pred_answer', pred_answer)

        return answers_probs, pred_answer