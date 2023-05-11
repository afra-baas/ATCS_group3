from typing import List

from src.prompts.prompt import Prompt

class NLIPrompt(Prompt):
    prompt_instructions = ""
    prompt_query = ""

    def __call__(self, sentences: List[str]) -> str:
        # :param sentences: a list with all the input sentences
        # :return: a string transformed to the desired prompt.
        prompt = f"{sentences[0]} \n Question: {sentences[1]} True, False, or Neither?"
        return prompt
