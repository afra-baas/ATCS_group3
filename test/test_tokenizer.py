# import pytest
from transformers import AutoTokenizer


# @pytest.fixture
def tokenizer(LM_model):
    return AutoTokenizer.from_pretrained(LM_model)


def test_tokenization_shape(tokenizer, LM_model):
    words = ["Yes", "No", "Maybe", "True", "False", "Neither", "A"]
    input_ids = tokenizer(words)["input_ids"]
    for i, id in enumerate(input_ids):
        if LM_model == 'huggyllama/llama-7b' and len(id) == 2:
            # print('word:', words[i], 'id:', id)
            id = [id[1]]
        elif LM_model == 'bigscience/T0pp' and len(id) == 2:
            # print('word:', words[i], 'id:', id)
            id = [id[0]]

        assrt = '' if len(id) == 1 else 'assert error'
        print('word:', words[i], 'id:', id, assrt)
        # assert len(id) == 1


if __name__ == "__main__":
    # LM_model = 'huggyllama/llama-7b' #non-instruction tuned
    # LM_model = 'bigscience/bloom-560m'  # non-instruction tuned
    # LM_model = 'bigscience/bloomz-560m'  # instruction tuned
    # LM_model = 'bigscience/T0pp'  # instruction tuned
    LM_models = ['huggyllama/llama-7b', 'bigscience/bloom-560m',
                 'bigscience/bloomz-560m', 'bigscience/T0pp']
    for LM_model in LM_models:
        print(f'------ {LM_model} ------')
        test_tokenization_shape(tokenizer(LM_model), LM_model)
