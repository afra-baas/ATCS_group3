from typing import List

from prompts.prompt import Prompt


class NLIPrompt(Prompt):
    prompt_instructions = ""
    prompt_query = ""

    def __call__(self, sentences: List[str]) -> str:
        # :param sentences: a list with all the input sentences
        # :return: a string transformed to the desired prompt.
        # prompt = f"{sentences[0]} \n Question: {sentences[1]} true, false, or neither?"
        prompt = f"{sentences[0]} \n Question: {sentences[1]} \n Possible answers: \n yes:entailment \n no:contradiction \n maybe: neutral"
        return prompt
