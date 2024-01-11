import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


from data_utils.data_utils_ER import IEMOCAPERProcessor

logger = logging.getLogger(__name__)

class FTProcessor(Dataset):
    def __init__(self, data_type):
            self.er_processor = IEMOCAPERProcessor("data")

    def __getitem__(self,idx):
        if idx < len(self.er_processor):
            return self.er_processor[idx]
        else:
            return self.er_processor[idx - len(self.asr_processor)]

    def __len__(self):
        return len(self.er_processor)
    
        

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    # return average word error rate
    return {"acc": (preds == labels).mean()}

    


def load_and_cache_examples(data_type,evaluate=False):
    return processors['finetune'](data_type)

processors = {
    "finetune": FTProcessor,
}

output_modes = {
    "audio": "text"
}