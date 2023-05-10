import pytest
import torch
from datasets import load_dataset

from src.data.NLI.dataset import NLIDataset

@pytest.fixture(scope="module")
def nli_dataset():
    dataset = [
        {"premise": "This is a premise", "hypothesis": "This is a hypothesis", "label": 1},
        {"premise": "Another premise", "hypothesis": "Another hypothesis", "label": 0},
        {"premise": "Third premise", "hypothesis": "Third hypothesis", "label": 2}
    ]
    return NLIDataset(dataset)

@pytest.fixture
def xnli_dataset():
    dataset = load_dataset("xnli", "fr")
    return NLIDataset(dataset['train'])

def test_nli_dataset_length(nli_dataset):
    assert len(nli_dataset) == 3

def test_nli_dataset_getitem(nli_dataset):
    premise, hypothesis, label = nli_dataset[0]
    assert premise == "This is a premise"
    assert hypothesis == "This is a hypothesis"
    assert label == 1

def test_xnli_dataset(xnli_dataset):
    assert len(xnli_dataset) > 0
    assert len(xnli_dataset[0]) == 3
    assert isinstance(xnli_dataset[0][0], str)
    assert isinstance(xnli_dataset[0][1], str)
    assert isinstance(xnli_dataset[0][2], int)