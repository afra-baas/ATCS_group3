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
            # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print('pad token added')

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Available device is {self.device}")

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print(f"Model device: {self.model.device}")

        self.possible_answers = possible_answers
        self.possible_answers_ids = [self.tokenizer(
            answer) for answer in self.possible_answers]
        self.one_hot_answers = []
        
        print(self.possible_answers_ids)
        test_token = self.tokenizer(
                    ["test"], return_tensors="pt", padding=True).to(self.device)
        if self.model_name == 'huggyllama/llama-7b':
            outputs = self.model(
                test_token["input_ids"], attention_mask=test_token["attention_mask"])
        else:
            outputs = self.model(**test_token, labels=test_token["input_ids"])
        output_size = outputs.logits.size()[-1]
        for answer in self.possible_answers_ids:
            answer = answer["input_ids"]
            if model_name == "llama":
                answer = answer[1:]
            one_hot = torch.zeros(size=(len(answer), output_size), device=self.device)
            for idx, id in enumerate(answer):
                one_hot[idx, id] = 1
            self.one_hot_answers.append(one_hot)
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
        full_sentence_probs = torch.zeros(size=(len(prompt), len(self.possible_answers)), device= self.device)
        with torch.no_grad():
            for idx, answer in enumerate(self.possible_answers):
                full_prompt = [p + " " + answer for p in prompt]
                inputs = self.tokenizer(
                    full_prompt, return_tensors="pt", padding=True).to(self.device)
                print('inputs: ', inputs)
                one_hot = self.one_hot_answers[idx]
                
                # generate outputs
                if self.model_name == 'huggyllama/llama-7b':
                    outputs = self.model(
                        inputs["input_ids"], attention_mask=inputs["attention_mask"])
                else:
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                answer_size = answer_size = 1+one_hot.size()[0]
                logits = outputs.logits[:, -answer_size : -1, :]
                softmax = torch.nn.Softmax(dim=2)
                probs = softmax(logits)
                probs += 1 * (10**-5) # we want no zeros for the log prob
                log_prob = torch.log(probs)
                answer_probs = torch.mul(log_prob, one_hot)
                final_probs = torch.sum(answer_probs, (1,2))
                # print(f"final probs : {final_probs}")
                # final_probs /= torch.mean(final_probs) * -1 # option 1
                softmax_2 = torch.nn.Softmax(dim=0) # option 2
                final_probs = softmax_2(final_probs)
                # print(f"final probs softmax {final_probs}") 
                # exit()
                full_sentence_probs[:, idx] = final_probs
            # # generate outputs
            # outputs = self.model(
            #     inputs["input_ids"], attention_mask=inputs["attention_mask"])

           
        print('answers_probs:', full_sentence_probs)
        pred_answer_indices = full_sentence_probs.argmax(dim=1)
        pred_answer = [self.possible_answers[i] for i in pred_answer_indices]
        torch.cuda.empty_cache()
        return full_sentence_probs, pred_answer
