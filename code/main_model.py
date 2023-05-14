from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import torch


class Classifier():
    def __init__(self, model_name, device="cpu"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda" if device == "cuda" else "cpu")
        print('self.device ', self.device)

        # "bigscience/bloom-560m"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, device=self.device)
        if model_name == 'xlm-roberta-base':
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)

    def __call__(self, prompt, possible_answers):
        """
        Generate probabilites for each promt and each possible answer.
        Input: 
            prompt: list of strings where each string is a seperate prompt
            possible_answers: list of strings where each string is an answer we want the logits from

        Output:
            answer_probs: tensor of shape (len(prompt), len(possible_answer)) where the values are the logits for each answer per prompt
        """

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # tokenize input and possible answers
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True).to(self.device)
        possible_answers_ids = [self.tokenizer(
            answer) for answer in possible_answers]
        print(possible_answers)
        exit()
        # generate outputs
        if  torch.cuda.is_available():
            print('summary: ', torch.cuda.memory_summary(device=self.device))
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

        torch.cuda.empty_cache()
        return answers_probs, pred_answer


# if __name__ == "__main__":

#     LM_model = 'bigscience/bloom-560m'
#     model = Classifier(LM_model)
#     prompt = ['what is the sentiment of this review: i like trains a lot',
#               'is the sentiment of this review positive or negative? i like trains']
#     possible_answers = ['positive', 'negative']
#     answers_probs, pred_answer = model(prompt, possible_answers)
#     print(answers_probs, pred_answer)
