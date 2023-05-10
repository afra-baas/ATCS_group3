import pytest

from src.data.MARC.dataset import MARCDataset

@pytest.fixture
def marc_dataset():
    return MARCDataset().dataset

def test_marc_dataset(marc_dataset):
    train_dataset = marc_dataset['train']
    assert len(train_dataset) > 0
    assert len(train_dataset[0]) == 8

