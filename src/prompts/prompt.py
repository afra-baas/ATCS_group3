from typing import List


class Prompt:
    """
    Generates a promt for every sentence according to the instructions provided
    """

    prompt_instructions = ""  # a str with the general prompt instructions.
    prompt_query = ""  # a str with the question we want to ask the language model

    def __call__(self, sentences: List[str]) -> str:
        # :param sentences: a list with all the input sentences
        # :return: a string transformed to the desired prompt.
        raise NotImplementedError
