from models.model import Model

test_sentence = ["this is a test"]
answers = ["a", "b"]
llama_model = Model("llama")
llama_model(test_sentence, answers)
