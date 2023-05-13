import pytest
import torch
from transformers import AutoTokenizer
from src.models.model import Model


# Define fixtures
@pytest.fixture()
def model():
    model_name = "roberta"
    return Model(model_name)


@pytest.fixture()
def prompt():
    return ["What is the capital of France?", "Who is the author of The Great Gatsby?"]


@pytest.fixture()
def possible_answers():
    return ["Paris", "London", "New York", "F. Scott Fitzgerald"]


# Test cases
def test_model_output_shape(model, prompt, possible_answers):
    answer_probs, pred_answer = model(prompt, possible_answers)
    assert answer_probs.shape == (len(prompt), len(possible_answers))


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
