import pytest
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data.NLI.dataset import NLIDataset
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
        assert isinstance(batch, dict)
        assert len(batch.keys()) == 3
        # Batch size is correct
        assert len(batch['premise']) == len(batch['hypothesis']) == len(batch['label']) == 32
        assert all(isinstance(x, str) for x in batch['premise'])
        assert all(isinstance(x, str) for x in batch['hypothesis'])
