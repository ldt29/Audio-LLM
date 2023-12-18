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

class ERProcessor(IEMOCAP):
    def __init__(self, data_dir, sessions):
        super().__init__(data_dir, sessions)
        self.prompts = ['Please give me the emotion of this speech. The emotion of this speech is ',
                        'What is the emotion of this speech? The emotion of this speech is',
                        'Please tell me the emotion of this speech. The emotion of this speech is',
                        'I want to know the emotion of this speech. The emotion of this speech is',
                        'Tell me the emotion of this speech. The emotion of this speech is']

    def __getitem__(self,idx):
        metadata = self.get_metadata(idx)
        return str(self._path)+'/'+metadata[0], get_IEMOCAP_9target(metadata[3]), self.prompts[idx%len(self.prompts)]


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    # return average word error rate
    return {"acc": (preds == labels).mean()}

def load_and_cache_examples(args, data_type, evaluate=False):
    if data_type=='train':
        processor = processors["wave2text"](args.train_data_dir,('1', '2', '3', '4'))
    elif data_type=='dev':
        processor = processors["wave2text"](args.dev_data_dir,('5'))
    else:
        processor = processors["wave2text"](args.test_data_dir,('5'))

    return processor




processors = {
    "wave2text": ERProcessor,
}

output_modes = {
    "wave2text": "text"
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