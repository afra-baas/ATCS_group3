import pytest
from torch.utils.data import DataLoader
from src.data.NLI.dataloader import NLIDataLoader

@pytest.fixture(scope="module")
def nli_dataloader():
    return NLIDataLoader("fr", batch_size=32)

def test_nli_dataloader(nli_dataloader):
    dataloader = nli_dataloader.get_dataloader()
    assert isinstance(dataloader, DataLoader)
    assert len(dataloader) > 0

    # Test if data is loaded correctly
    for batch in dataloader:
        assert isinstance(batch, list)
        assert len(batch) == 32
        assert isinstance(batch[0], tuple)
        assert isinstance(batch[0][0], str)
        assert all(isinstance(x[0][0], str) for x in batch)
        assert all(isinstance(x[0][1], str) for x in batch)
        break
