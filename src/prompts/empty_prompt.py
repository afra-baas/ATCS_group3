from typing import List

from prompts.prompt import Prompt


class EmptyPrompt(Prompt):
    prompt_instructions = ""
    prompt_query = ""

    def __call__(self, sentences: List[str]) -> str:
        # :param sentences: a list with all the input sentences
        # :return: a string transformed to the desired prompt.
        return sentences[0]
