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


from data_utils.data_utils_ASR import LibriSpeechASRProcessor, COVOSTASRProcessor 


logger = logging.getLogger(__name__)

class ASRProcessor(Dataset):
    def __init__(self, data_type):
        self.processor_1 = LibriSpeechASRProcessor(data_type)
        self.processor_2 = COVOSTASRProcessor(data_type)

    def __getitem__(self,idx):
        if idx < len(self.processor_1):
            return self.processor_1[idx]
        else:
            return self.processor_2[idx - len(self.processor_1)]

    def __len__(self):
        return len(self.processor_1)
    
        

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    # return average word error rate
    return {"acc": (preds == labels).mean()}

    


def load_and_cache_examples(data_type):

    return processors["asr"](data_type)

processors = {
    "asr": ASRProcessor,

}

output_modes = {
    "asr": "text"
}