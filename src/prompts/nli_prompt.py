from typing import List
from prompts.prompt import Prompt


class NLIPrompt(Prompt):

    def __call__(self, sentences: List[str], prompt_type, prompt_id) -> str:
        # :param sentences: a list with all the input sentences ## ??
        # :return: a string transformed to the desired prompt.

        template = self.dict_sa_prompt[prompt_type][prompt_id]
        premise = sentences[0]
        hypothesis = sentences[1]
        prompt = template.format(premise=premise, hypothesis=hypothesis)
        return prompt
