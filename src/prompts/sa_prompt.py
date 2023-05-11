from typing import List

from src.prompts.prompt import Prompt


class SAPrompt(Prompt):
    prompt_instructions = "Can you please tell me the sentiment of this review"
    prompt_query = "is it positive or nagative?"

    def __call__(self, sentences: List[str]) -> str:
        # :param sentences: a list with all the input sentences
        # :return: a string transformed to the desired prompt.
        prompt = (
            "We will give you a set of instructions an input sentence and a querry. "
        )
        prompt += "You should answer the querry based on the input sentence accord to the instructions provided. \n"
        prompt += f"instructions: {self.prompt_instructions} \ninput sentence: {sentences[0]} \nquerry: {self.prompt_query} \nanswer: "
        return prompt
