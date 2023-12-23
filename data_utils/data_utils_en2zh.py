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
import csv

import numpy as np
import torch
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class COVOSTEn2ZhProcessor(Dataset):
    # get audio file name and text
  
    def __init__(self,data_type):
        self.clips_path = 'data/covost/cv-corpus-15.0-2023-09-08/en/clips/'
        self.tsv_file_path = 'data/covost/cv-corpus-15.0-2023-09-08/en/covost_v2.en_zh-CN.' + data_type + '.tsv'
        self.audio_text_list = pd.read_csv(self.tsv_file_path, sep="\t", header=0, encoding="utf-8",
                       escapechar="\\", quoting=csv.QUOTE_NONE, na_filter=False)
        self.prompts = ['This front one is speech audio. When this speech audio is converted to text and then translated to Chinese, it is that']

    
    def __len__(self):
        return len(self.audio_text_list)
    
    def __getitem__(self, idx):
        return  self.clips_path + self.audio_text_list['path'][idx], self.audio_text_list['translation'][idx], self.prompts[idx%len(self.prompts)]


def compute_en2zh_metrics(preds, labels):
    assert len(preds) == len(labels)
    # return average word error rate
    return {"acc": (preds == labels).mean()}

    


def load_and_cache_en2zh_examples(data_type):
    return processors["en2zh"](data_type)


processors = {
    "en2zh": COVOSTEn2ZhProcessor,
}

output_modes = {
    "en2zh": "text"
}

