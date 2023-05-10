from typing import List


class Prompt:
    """
    Generates a promt for every sentence according to the instructions provided
    """

    def __init__(self, prompt_instructions: str, prompt_querry: str):
        # :param prompt_instructions: a str with the general prompt instructions.
        # :param prompt_querry: a str with the question we want to ask the language model
        self.prompt_instructions = prompt_instructions
        self.prompt_querry = prompt_querry

    def __call__(self, sentences: List[str]) -> List[str]:
        # :param sentences: a list with all the input sentences
        # :return: a list with all sentences transformed to the desired prompt.
        output = []
        for sentence in sentences:
            promt = "We will give you a set of instructions an input sentence and a querry. "
            promt += "You should answer the querry based on the input sentence accord to the instructions provided. \n"
            promt += f"instructions: {self.prompt_instructions} \ninput sentence: {sentence} \nquerry: {self.prompt_querry} \nanswer: "
            output.append(promt)
        return output
