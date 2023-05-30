from typing import List
from main import prompt_templates


class Prompt:
    """
    Generates a promt for every sentence according to the instructions provided
    """

    def __init__(self, language, task):
        self.language = language
        self.task = task
        self.dict_sa_prompt = prompt_templates[language][task]

    def __call__(self, sentences: List[str]) -> str:
        # :param sentences: a list with all the input sentences
        # :return: a string transformed to the desired prompt.
        raise NotImplementedError
