import pytest
import torch
from transformers import AutoTokenizer

from config import data
from src.data.NLI.dataloader import NLIDataLoader
from src.models.model import Model

# Define fixtures
@pytest.fixture()
def model():
    model_name = "xlm-roberta-base"
    return Model(model_name)

@pytest.fixture(scope="module")
def nli_dataloader():
    return NLIDataLoader("fr", batch_size=32).get_dataloader()

# Test cases
def test_model_output_shape(model, nli_dataloader):
    for batch in nli_dataloader:
        prompt = batch[0][0]
        possible_answers = list(data["NLI"]["label_to_meaning"].values())
        values = model([prompt], possible_answers)
        print()

def test_model_device(model):
    assert str(model.device).startswith("cuda") or str(model.device) == "cpu"

def test_model_output_type(model, prompt, possible_answers):
    answer_probs, pred_answer = model(prompt, possible_answers)
    assert isinstance(answer_probs, torch.Tensor)
    assert isinstance(pred_answer, list)

def test_model_output_values(model, prompt, possible_answers):
    answer_probs, pred_answer = model(prompt, possible_answers)
    assert (answer_probs >= 0).all() and (answer_probs <= 1).all()

def test_model_tokenizer(model):
    assert isinstance(model.tokenizer, AutoTokenizer)

def test_model_raises_error_for_unsupported_model():
    with pytest.raises(KeyError):
        model = Model("invalid_model_name")