import sys
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchaudio.datasets.iemocap import IEMOCAP

logger = logging.getLogger(__name__)

def get_IEMOCAP_9target(target):
    # 'ang':  anger 
    # 'hap':  happiness 
    # 'exc':  excitement 
    # 'sad':  sadness
    # 'fru':  frustration 
    # 'fea':  fear 
    # 'sur':  surprise 
    # 'neu':  neutral state 
    # 'xxx':  other 

    if (target == "ang"):
        target = "angry"
    elif (target == "hap"):
        target = "happy"
    elif (target == "exc"):
        target = "excited"
    elif (target == "sad"):
        target = "sad"
    elif (target == "fru"):
        target = "frustrated"
    elif (target == "fea"):
        target = "fear"
    elif (target == "sur"):
        target = "surprised"
    elif (target == "neu"):
        target = "neutral"
    elif (target == "xxx"):
        target = "other"
    else:
        print(target)
        target=None
        print("Error, skip!")

    return target

class IEMOCAPERProcessor(IEMOCAP):
    def __init__(self, data_type):
        if data_type=='train':
            sessions = ('1', '2', '3', '4')
        else:
            sessions = ('5')
        super().__init__('data', sessions)
        self.prompts = ['The front one is speech audio, and the emotion of this speech is']

    def __getitem__(self,idx):
        metadata = self.get_metadata(idx)
        return str(self._path)+'/'+metadata[0], get_IEMOCAP_9target(metadata[3]), self.prompts[idx%len(self.prompts)]


def compute_er_metrics(preds, labels):
    assert len(preds) == len(labels)
    # return average word error rate
    return {"acc": (preds == labels).mean()}

def load_and_cache_er_examples(data_type):

    return processors["er"](data_type)




processors = {
    "er": IEMOCAPERProcessor,
}

output_modes = {
    "er": "text"
}


# def get_metadata(self, n: int) -> Tuple[str, int, str, str, str]:
#         """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
#         but otherwise returns the same fields as :py:meth:`__getitem__`.

#         Args:
#             n (int): The index of the sample to be loaded

#         Returns:
#             Tuple of the following items;

#             str:
#                 Path to audio
#             int:
#                 Sample rate
#             str:
#                 File name
#             str:
#                 Label (one of ``"neu"``, ``"hap"``, ``"ang"``, ``"sad"``, ``"exc"``, ``"fru"``)
#             str:
#                 Speaker
#         """