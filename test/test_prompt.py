# Test Prompt
from src.prompts.prompt import Prompt

def test_Prompt():
    prompt_instructions = "Classify the sentiment of the sentence as positive, negative, or neutral."
    prompt_querry = "What is the sentiment of the sentence?"
    prompt = Prompt(prompt_instructions, prompt_querry)

    sentences = ["I love this movie", "I hate this restaurant", "This book is okay"]
    expected_output = [
        "We will give you a set of instructions an input sentence and a querry. You should answer the querry based on the input sentence accord to the instructions provided. \ninstructions: Classify the sentiment of the sentence as positive, negative, or neutral. \ninput sentence: I love this movie \nquerry: What is the sentiment of the sentence? \nanswer: ",
        "We will give you a set of instructions an input sentence and a querry. You should answer the querry based on the input sentence accord to the instructions provided. \ninstructions: Classify the sentiment of the sentence as positive, negative, or neutral. \ninput sentence: I hate this restaurant \nquerry: What is the sentiment of the sentence? \nanswer: ",
        "We will give you a set of instructions an input sentence and a querry. You should answer the querry based on the input sentence accord to the instructions provided. \ninstructions: Classify the sentiment of the sentence as positive, negative, or neutral. \ninput sentence: This book is okay \nquerry: What is the sentiment of the sentence? \nanswer: "
    ]
    assert prompt(sentences) == expected_output