from transformers import AutoTokenizer, LlamaTokenizer
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

        print(f"Loading model {self.model_name}")
        if model_name == "llama" or model_name == "alpaca":
            self.model = model_config["model_constructor"](
                self.model_name, torch_dtype=torch.float16)
        else:
            self.model = model_config["model_constructor"](self.model_name)
        print(f"Model {self.model_name} loaded")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    self.model_name)
            except Exception as e:
                # Handle the exception here
                print("An error occurred while creating the tokenizer:", str(e))

        if model_name == 'llama':
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print('pad token added')

        elif model_name == 'alpaca':
            self.tokenizer.pad_token = " "
            print('pad token added')

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Available device is {self.device}")

        # print('summary: ', torch.cuda.memory_summary(device=self.device))
        # self.model.to(self.device)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print(f"Model device: {self.model.device}")

    def __call__(self, prompt: List[str], possible_answers, language):
        """
        Generate probabilites for each promt and each possible answer.
        :param prompt: a list of strings with the prompts
        :param possible_answers: a list of strings with the possible answers
        :return: a tensor of shape (len(prompt), len(possible_answer))  where the values are the logits for each answer per prompt
        """

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True).to(self.device)
        possible_answers_ids = [self.tokenizer(
            answer) for answer in possible_answers]

        # generate outputs
        # print('summary: ', torch.cuda.memory_summary(device=self.device))
        with torch.no_grad():
            if self.model_name == 'huggyllama/llama-7b' or self.model_name == 'chainyo/alpaca-lora-7b':
                outputs = self.model(
                    inputs["input_ids"], attention_mask=inputs["attention_mask"])
            else:
                outputs = self.model(**inputs, labels=inputs["input_ids"])

        # get the logits of the last token
        logits = outputs.logits[:, -1]
        logits = torch.nn.functional.softmax(logits, dim=1)

        # loop over all possible answers for every promt and store the logits
        answers_probs = torch.zeros(len(prompt), len(possible_answers_ids)).to(
            self.device)

        for idx, answer in enumerate(possible_answers_ids):
            print('answer ', answer)
            if len(answer) == 0:
                print('len answer was 0')
                continue

            # TO DO: check if this is the correct shoter version
            # id = answer["input_ids"]
            # probs = []
            # if language == 'en':
            #     id = [id[1]] if self.model_name in ['huggyllama/llama-7b', 'chainyo/alpaca-lora-7b'] and len(id) == 2 else id
            #     probs = [logits[:, id]] if len(id) == 1 else [logits[:, [part]] for part in id]
            # else:
            #     id = [id[1]] if self.model_name in ['huggyllama/llama-7b', 'chainyo/alpaca-lora-7b'] and len(id) == 2 else id
            #     probs = [logits[:, id]] if len(id) == 1 else [logits[:, [part]] for part in id]
            # answers_probs[:, idx] = torch.cat(probs, dim=1).mean(dim=1).T

            if language == 'en':
                id = answer["input_ids"]
                if self.model_name == 'huggyllama/llama-7b' and len(id) == 2:
                    print(f'id: {id} -> {[id[1]]}')
                    id = [id[1]]
                    probs = logits[:, id]
                    print('probs_ shape', probs.shape)
                    answers_probs[:, idx] = probs.T
                    print(f'id: {id} -> {probs}/{probs.T}')
                elif self.model_name == 'google/flan-t5-base' and len(id) == 2:
                    print(f'id: {id} -> {[id[0]]}')
                    id = [id[0]]
                    probs = logits[:, id]
                    print('probs_ shape', probs.shape)
                    answers_probs[:, idx] = probs.T
                    print(f'id: {id} -> {probs}/{probs.T}')

                elif len(id) > 1:
                    # TO DO: test if this is the best solution
                    probs = []
                    for part in id:
                        part_id = [part]
                        probs.append(logits[:, part_id])
                    probs_ = torch.cat(
                        probs, dim=1).mean(dim=1)
                    print('probs_ shape', probs_.shape)
                    # answers_probs[:, idx] = probs_.T
                    # answers_probs[:, idx] = probs_.transpose(0, 1)
                    answers_probs[:, idx] = probs_
                    # print(f'id: {id} -> {probs_.transpose(0, 1)}, {(probs_.transpose(0, 1)).shape}')
                    print(f'id: {id} -> {probs_}, {(probs_).shape}')
            else:
                id = answer["input_ids"]
                if self.model_name == 'huggyllama/llama-7b' and len(id) == 2:
                    print(f'id: {id} -> {[id[1]]}')
                    id = [id[1]]
                    probs = logits[:, id]
                    print('probs_ shape', probs.shape)
                    answers_probs[:, idx] = probs.T
                    print(f'id: {id} -> {probs}/{probs.T}')
                elif self.model_name == 'google/flan-t5-base' and len(id) == 2:
                    print(f'id: {id} -> {[id[0]]}')
                    id = [id[0]]
                    probs = logits[:, id]
                    print('probs_ shape', probs.shape)
                    answers_probs[:, idx] = probs.T
                    print(f'id: {id} -> {probs} / {probs.T}')

                elif len(id) > 1:
                    # TO DO: test if this is the best solution
                    probs = []
                    for part in id:
                        part_id = [part]
                        probs.append(logits[:, part_id])
                    probs_ = torch.cat(
                        probs, dim=1).mean(dim=1)
                    print('probs_ shape', probs_.shape)
                    # answers_probs[:, idx] = probs_.T
                    # answers_probs[:, idx] = probs_.transpose(0, 1)
                    answers_probs[:, idx] = probs_
                    # print(f'id: {id} -> {probs_.transpose(0, 1)}, {(probs_.transpose(0, 1)).shape}')
                    print(f'id: {id} -> {probs_}, {(probs_).shape}')
                # elif len(id) > 1:
                #     # TO DO: test if this is the best solution
                #     id = torch.tensor(id).to(self.device)
                #     probs = torch.index_select(logits, dim=1, index=id)
                #     answers_probs[:, idx] = probs.T.squeeze(1)
                #     print(f'id: {id} -> {probs.T} -> {probs.T.squeeze(1)}')

        print('answers_probs:', answers_probs)
        pred_answer_indices = answers_probs.argmax(dim=1)
        pred_answer = [possible_answers[i] for i in pred_answer_indices]

        torch.cuda.empty_cache()
        return answers_probs, pred_answer
