import pytest
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data.NLI.dataset import NLIDataset
from src.data.NLI.dataloader import NLIDataLoader

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

def test_nli_dataloader(nli_dataset):
    dataloader = NLIDataLoader(dataset=nli_dataset, batch_size=32).get_dataloader()
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) > 0

    # Test if data is loaded correctly
    for batch in dataloader:
        assert isinstance(batch, tuple)
        assert len(batch) == 3
        assert isinstance(batch[0], list)
        assert isinstance(batch[1], list)
        assert isinstance(batch[2], list)
        assert len(batch[0]) == len(batch[1]) == len(batch[2]) == 32
        assert all(isinstance(x, str) for x in batch[0])
        assert all(isinstance(x, str) for x in batch[1])
        assert all(isinstance(x, int) for x in batch[2])