import pytest

from src.data.NLI.dataset import NLIDataset

@pytest.fixture
def nli_dataset():
    return NLIDataset().dataset

def test_xnli_dataset(nli_dataset):
    train_dataset = nli_dataset['train']
    assert len(train_dataset) > 0
    assert len(train_dataset[0]) == 3
    assert isinstance(train_dataset[0]['hypothesis'], str)
    assert isinstance(train_dataset[0]['premise'], str)
