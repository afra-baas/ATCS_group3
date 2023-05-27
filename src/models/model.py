from transformers import AutoTokenizer, LlamaTokenizer
import torch
from typing import List
from config import model


class Model:
    def __init__(self, model_name: str, possible_answers: list):
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
        if model_name == "llama":
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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Available device is {self.device}")

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print(f"Model device: {self.model.device}")

        self.possible_answers = possible_answers
        possible_answers_ids_before = [self.tokenizer(
            answer) for answer in self.possible_answers]
        self.possible_answers_ids = []
        for answer in possible_answers_ids_before:
            print('answer ', answer)
            if len(answer) == 0:
                print('len answer was 0')
                continue

            id = answer["input_ids"]
            if self.model_name == 'huggyllama/llama-7b' and len(id) == 2:
                print(f'id: {id} -> {[id[1]]}')
                id = [id[1]]
            elif self.model_name == 'google/flan-t5-base' and len(id) == 2:
                print(f'id: {id} -> {[id[0]]}')
                id = [id[0]]
            else:
                print(f'id:{id}')
            self.possible_answers_ids.append(id)

    def __call__(self, prompt: List[str]):
        """
        Generate probabilites for each promt and each possible answer.
        :param prompt: a list of strings with the prompts
        :param possible_answers: a list of strings with the possible answers
        :return: a tensor of shape (len(prompt), len(possible_answer))  where the values are the logits for each answer per prompt
        """

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        with torch.no_grad():

            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True).to(self.device)
            # print('inputs: ', inputs)

            # generate outputs
            if self.model_name == 'huggyllama/llama-7b':
                outputs = self.model(
                    inputs["input_ids"], attention_mask=inputs["attention_mask"])
            else:
                outputs = self.model(**inputs, labels=inputs["input_ids"])

            # # generate outputs
            # outputs = self.model(
            #     inputs["input_ids"], attention_mask=inputs["attention_mask"])

            # get the logits of the last token
            logits = outputs.logits[:, -1]
            logits = torch.nn.functional.softmax(logits, dim=1)

            # loop over all possible answers for every promt and store the logits
            answers_probs = torch.zeros(len(prompt), len(self.possible_answers_ids)).to(
                self.device)

            for idx, answer_id in enumerate(self.possible_answers_ids):

                # avg version
                # if len(answer_id) > 1:
                #     # TO DO: test if this is the best solution
                #     probs = []
                #     for part in answer_id:
                #         part_id = [part]
                #         probs.append(logits[:, part_id])
                #     probs_ = torch.cat(
                #         probs, dim=1).mean(dim=1)

                #     print('probs_ shape', probs_, (probs_).shape)
                #     # answers_probs[:, idx] = probs_.T
                #     answers_probs[:, idx] = probs_
                #     # print(f'id: {answer_id} -> {probs_}, {(probs_).shape}')

                # summ
                print('summ of probs approach')
                if len(answer_id) > 1:
                    # TO DO: test if this is the best solution
                    probs = []
                    for part in answer_id:
                        part_id = [part]
                        probs.append(logits[:, part_id])
                    probs_ = torch.cat(
                        probs, dim=1).sum(dim=1)

                    print('probs_ shape', probs_, (probs_).shape)
                    # answers_probs[:, idx] = probs_.T
                    answers_probs[:, idx] = probs_
                    # print(f'id: {answer_id} -> {probs_}, {(probs_).shape}')

                # # max
                # print('max of probs approach')
                # if len(answer_id) > 1:
                #     # TO DO: test if this is the best solution
                #     probs = []
                #     for part in answer_id:
                #         part_id = [part]
                #         probs.append(logits[:, part_id])
                #     max_prob, max_token_id = torch.max(
                #         torch.cat(probs, dim=1), dim=1)
                #     probs_ = max_prob
                #     print('probs_ shape', probs_, (probs_).shape)
                #     # answers_probs[:, idx] = probs_.T
                #     answers_probs[:, idx] = probs_
                #     print(f'id: {answer_id} -> {probs_}, {(probs_).shape}')

                else:
                    probs = logits[:, answer_id]
                    answers_probs[:, idx] = probs.T
                    print(f'id: {answer_id} -> {probs.T}, {(probs.T).shape}')

        # print('answers_probs:', answers_probs)
        pred_answer_indices = answers_probs.argmax(dim=1)
        pred_answer = [self.possible_answers[i] for i in pred_answer_indices]

        torch.cuda.empty_cache()
        return answers_probs, pred_answer
