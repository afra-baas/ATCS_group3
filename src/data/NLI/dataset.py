from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch.nn.utils.rnn as rnn_utils

from src.data.hf_dataset import HFDataset


class NLIDataset(HFDataset):
    language = "fr"
    dataset_name = "xnli"
