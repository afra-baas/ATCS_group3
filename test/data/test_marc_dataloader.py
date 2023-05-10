import pytest
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.data.MARC.dataloader import MARCDataLoader

@pytest.fixture(scope="module")
def marc_dataloader():
    return MARCDataLoader("fr", batch_size=32)

def test_marc_dataloader(marc_dataloader):
    dataloader = marc_dataloader.get_dataloader()
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) > 0

    # Test if data is loaded correctly
    for batch in dataloader:
        assert isinstance(batch, list)
        assert len(batch) == 32
        assert isinstance(batch[0], tuple)
        assert len(batch[0][0]) == 1
        assert all(isinstance(x[0][0], str) for x in batch)
        break
