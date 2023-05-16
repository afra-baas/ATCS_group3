from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import torch
from typing import List
import os
from config import model
from datetime import datetime


class Model:
    def __init__(self, model_name: str):
        """
        :param model_name: name of the model to use
        :param device: device to use
        """
        try:
            model_config = model["SUPPORTED_MODELS"][model_name]
        except KeyError:
            raise KeyError(
                f"Model {model_name} not supported. Please use one of the following models: {model['SUPPORTED_MODELS'].keys()}"
            )
        self.model_name = model_config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if model_name == 'llama' or model_name == 'alpaca':
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model {self.model_name}")
        self.model = model_config["model_constructor"](self.model_name)
        print(f"Model {self.model_name} loaded")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # print('summary: ', torch.cuda.memory_summary(device=self.device))
        # self.model.to(self.device)
        print(f"Moving model to {self.device}")

    def __call__(self, prompt: List[str], possible_answers):
        """
        Generate probabilites for each promt and each possible answer.
        :param prompt: a list of strings with the prompts
        :param possible_answers: a list of strings with the possible answers
        :return: a tensor of shape (len(prompt), len(possible_answer))  where the values are the logits for each answer per prompt
        """

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # tokenize input and possible answers
        # , add_special_tokens=False)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        # inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        # print('inputs ', inputs)

        possible_answers_ids = [self.tokenizer(
            answer) for answer in possible_answers]  # , add_special_tokens=False)

        # generate outputs
        # print('summary: ', torch.cuda.memory_summary(device=self.device))
        if self.model_name == 'huggyllama/llama-7b' or self.model_name == 'chainyo/alpaca-lora-7b':
            outputs = self.model(
                inputs["input_ids"], attention_mask=inputs["attention_mask"])
        else:
            outputs = self.model(**inputs, labels=inputs["input_ids"])

        # get the logits of the last token
        logits = outputs.logits[:, -1]

        # loop over all possible answers for every promt and store the logits
        answers_probs = torch.zeros(len(prompt), len(possible_answers_ids)).to(
            self.device)

        for idx, answer in enumerate(possible_answers_ids):
            print('answer ', answer)
            if len(answer) == 0:
                print('len answer was 0')
                continue

            id = answer["input_ids"]
            if self.model_name == 'huggyllama/llama-7b' and len(id) == 2:
                print(f'id: {id} -> {[id[1]]}')
                id = [id[1]]
            elif self.model_name == 'chainyo/alpaca-lora-7b' and len(id) == 2:
                print(f'id: {id} -> {[id[1]]}')
                id = [id[1]]
            elif self.model_name == 'google/flan-t5-base' and len(id) == 2:
                print(f'id: {id} -> {[id[0]]}')
                id = [id[0]]
            probs = logits[:, id]
            answers_probs[:, idx] = probs.T

            # Save logits to a text file
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task = 'SA' if len(possible_answers) == 2 else 'NLI'
            log_file = f"saved_logits/{self.model_name}__{task}__{len(prompt)}.txt"

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            with open(log_file, "w+") as f:
                for idx, answer in enumerate(possible_answers_ids):
                    f.write(f"Answer {idx}:\n")
                    f.write(f"Answer ID: {answer['input_ids']}\n")
                    f.write(f"Logits: {answers_probs[:, idx]}\n\n")
            print(f"Logits saved to {log_file}")

        # pred_answer = possible_answers[answers_probs.index(max(answers_probs))]
        pred_answer_indices = answers_probs.argmax(dim=1)
        pred_answer = [possible_answers[i] for i in pred_answer_indices]
        # print("pred_answer", pred_answer)

        torch.cuda.empty_cache()
        return answers_probs, pred_answer
