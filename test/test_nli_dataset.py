import pytest

from src.data.NLI.dataset import NLIDataset

@pytest.fixture
def nli_dataset():
    return NLIDataset()

def test_xnli_dataset(nli_dataset):
    train_dataset = nli_dataset['train']
    assert len(train_dataset) > 0
    assert len(train_dataset) == 3
    assert isinstance(train_dataset[0][0], str)
    assert isinstance(train_dataset[0][1], str)
