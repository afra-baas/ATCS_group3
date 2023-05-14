import pytest
import torch
from transformers import AutoTokenizer

from src.data.MARC.dataloader import MARCDataLoader
from src.models.model import Model
from src.config import model as model_config
# from ATCS_group3.src.config import model as model_config

# Define fixtures


@pytest.fixture()
def model():
    model_name = "bloom"
    return Model(model_name)


@pytest.fixture(scope="module")
def prompt():
    dataloader = MARCDataLoader("en", batch_size=32).get_dataloader()
    for batch in dataloader:
        prompt = batch[0][0]
        break
    return prompt


@pytest.fixture(scope="module")
def marc_dataloader():
    return MARCDataLoader("en", batch_size=32)


@pytest.fixture(scope="module")
def possible_answers():
    return ['yes', 'no']


def test_tokenizer_correct_tokenization(possible_answers):
    for model_key, model_values in model_config['SUPPORTED_MODELS'].items():
        tokenizer = AutoTokenizer.from_pretrained(model_values['model_name'])
        for answer in possible_answers:
            tokenization = tokenizer(answer)
            assert len(tokenization.data['input_ids']) == 1

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
