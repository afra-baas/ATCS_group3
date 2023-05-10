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
    return NLIDataset()

def test_xnli_dataset(xnli_dataset):
    train_dataset = xnli_dataset['train']
    assert len(train_dataset) > 0
    assert len(train_dataset[0]) == len(train_dataset[1])
    assert isinstance(train_dataset[0][0], str)
    assert isinstance(train_dataset[1][0], str)
