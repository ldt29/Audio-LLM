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


from data_utils.data_utils_ASR import ASRProcessor
from data_utils.data_utils_ER import ERProcessor

logger = logging.getLogger(__name__)

class AudioProcessor(Dataset):
    def __init__(self, data_type):
        if data_type=='train':
            self.asr_processor = ASRProcessor("data/LibriSpeech/train-clean-100")
            self.er_processor = ERProcessor("data",('1', '2', '3', '4'))
        elif data_type=='dev':
            self.asr_processor = ASRProcessor("data/LibriSpeech/dev-clean")
            self.er_processor = ERProcessor("data",('5'))
        else:
            self.asr_processor = ASRProcessor("data/LibriSpeech/test-clean")
            self.er_processor = ERProcessor("data",('5'))

    def __getitem__(self,idx):
        if idx < len(self.asr_processor):
            return self.asr_processor[idx]
        else:
            return self.er_processor[idx - len(self.asr_processor)]

    def __len__(self):
        return len(self.asr_processor) + len(self.er_processor)
    
        

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    # return average word error rate
    return {"acc": (preds == labels).mean()}

    


def load_and_cache_examples(args,data_type,evaluate=False):
    if data_type=='train':
        processor = processors["audio"]('train')
    elif data_type=='dev':
        processor = processors["audio"]('dev')
    else:
        processor = processors["audio"]('test')

    return processor

processors = {
    "audio": AudioProcessor,

}

output_modes = {
    "audio": "text"
}