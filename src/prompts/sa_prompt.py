from typing import List

from prompts.prompt import Prompt


class SAPrompt(Prompt):
    prompt_instructions = 'Tell me if the sentiment of this review is positive'
    prompt_query = 'yes or no?'

    # dict_sa_prompt= {' passive': ['audiuebf', 'dbfksjbfke'], 'active':[ 'hdbfjweb', 'dbfsiugi']}
    # dict_sa_prompt= {'common modal verb': ['give me the sentiment', 'can you give me the sentiment ', 'would you give me the sentiment ', 'you ought to give me the sentiment']} #-> acc_base , acc_would , acc_can
    # dict_sa_prompt= {'common modal verb': ['give me the sentiment', 'can you give me the sentiment']} #-> acc_base , acc_can
    # dict_sa_prompt= {'common modal verb': ['give me the sentiment', 'would you give me the sentiment']} #-> acc_base , acc_would

    def __call__(self, sentences: List[str]) -> str:
        # :param sentences: a list with all the input sentences ## ??
        # :return: a string transformed to the desired prompt.
        prompt = (
            "We will give you a set of instructions an input sentence and a querry. "
        )
        prompt += "You should answer the querry based on the input sentence accord to the instructions provided. \n"
        prompt += f"instructions: {self.prompt_instructions} \ninput sentence: {sentences[0]} \nquerry: {self.prompt_query} \nanswer: "
        return prompt
