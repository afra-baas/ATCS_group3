from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


class Classifier():
    def __init__(self, model_name, device="cpu"):
        # 'xlm-roberta-base'
        # "bigscience/bloom-560m"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if device == "cuda" else "cpu")

    def __call__(self, prompt, possible_answers):

        # define the possible answers
        # possible_answers = ['positive', 'negative']

        # sample = 'I really liked this movie'
        # prompt = f"is this review positive or negative:  {sample}"

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        possible_answers_ids = [self.tokenizer.encode(
            answer) for answer in possible_answers]

        output = self.model(**inputs, labels=inputs["input_ids"])
        # print('output ', output)
        logits = output.logits[:, -1, :]

        # Get the probabilities of all tokens in the vocabulary
        probabilities = logits.softmax(dim=-1)
        # calculate the probabilities of possible answers
        answers_probs = []
        for idx, answer_id in enumerate(possible_answers_ids):
            probs = probabilities[0, answer_id[-1]].item()
            answers_probs.append(probs)
        # print('answers_probs', answers_probs)

        # # loop over all possible answers for every promt and store the logits
        # answers_probs = torch.zeros(len(prompt), len(
        #     possible_answers_ids)).to(self.device)
        # print('answers_probs ', answers_probs)
        # for idx, answer in enumerate(possible_answers_ids):
        #     id = answer["input_ids"]
        #     probs = logits[:, id][0]
        #     answers_probs[:, idx] = probs

        # determine the predicted answer
        pred_answer = possible_answers[answers_probs.index(max(answers_probs))]
        # print('pred_answer', pred_answer)

        # get the true answer
        # true_answer = 'positive' if labels.item() else 'negative'

        return answers_probs, pred_answer


# if __name__ == "__main__":

#     LM_model = 'xlm-roberta-base'
#     model = Classifier(LM_model)
#     prompt = 'what is the sentiment of this review: i like trains'
#     possible_answers = ['positive', 'negative']
#     answers_probs, pred_answer = model(prompt, possible_answers)
#     print(answers_probs, pred_answer)
