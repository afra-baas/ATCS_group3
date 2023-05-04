def prompt_generator(prompt_instructions, prompt_querry, sentences):
    """
    Generates a promt for every sentence according to the instructions provided
    input:
        prompt_instructions: a str with the general prompt instructions.
        prompt_querry: a str with the question we want to ask the language model
        sentences: a list with all the input sentences
    output:
        output: a list with all sentences transformed to the desired prompt.
    """
    output = []
    for sentence in sentences:
        promt = "We will give you a set of instructions an input sentence and a querry. "
        promt += "You should answer the querry based on the input sentence accord to the instructions provided. \n"
        promt += f"instructions: {prompt_instructions} \ninput sentence: {sentence} \nquerry: {prompt_querry} \nanswer: "
        output.append(promt)
    return output