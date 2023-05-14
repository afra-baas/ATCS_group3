from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch.nn.utils.rnn as rnn_utils

from data.hf_dataset import HFDataset


class NLIDataset(HFDataset):
    language = "en"
    dataset_name = "xnli"
