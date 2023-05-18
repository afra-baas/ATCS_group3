from models.model import Model

bloom_model = Model("bloom")
test_sentence = ["this is a test"]
answers = ["a", "b"]
bloom_model(test_sentence, answers)

llama_model = Model("llama")
llama_model(test_sentence, answers)
