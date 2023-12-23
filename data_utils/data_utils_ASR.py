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
import pandas as pd
import csv

logger = logging.getLogger(__name__)


class LibriSpeechASRProcessor(Dataset):
    # Audio Speech Recognition Processor
    # get audio file name and text
    # example:
    # data_dir/61/70970/61-70970.trans.txt
    # 61-70970-0005 THE LAD HAD CHECKED HIM THEN
    # 
    # audio file name: 61-70970-0005.flac
    # audio file path: data_dir/61/70970/61-70970-0005.flac
    # text: THE LAD HAD CHECKED HIM THEN
    def __init__(self, data_type):
        if data_type == 'train':
            data_dir = 'data/LibriSpeech/train-clean-360'
        elif data_type == 'dev':
            data_dir = 'data/LibriSpeech/dev-clean'
        else: 
            data_dir = 'data/LibriSpeech/test-clean'
        self.text_path_list = glob.glob(os.path.join(data_dir, '*', '*', '*.txt'))
        self.text_list = []
        self.audio_path_list = []
        self.prompts = ['This speech is saying that']
        
        for text_path in self.text_path_list:
            with open(text_path, 'r') as f:
                audio_path_pre = text_path.rsplit('/', 1)[0]
                for line in f.readlines():
                    line = line.strip()
                    if line != '':
                        self.text_list.append(line.split(' ', 1)[-1].lower())
                        audio_path = audio_path_pre + '/' + line.split(' ', 1)[0] + '.flac'
                        self.audio_path_list.append(audio_path)

    
        
    def __len__(self):
        return len(self.audio_path_list)
    
    def __getitem__(self, idx):
        return self.audio_path_list[idx], self.text_list[idx], self.prompts[idx%len(self.prompts)]


class COVOSTASRProcessor(Dataset):
    # get audio file name and text
  
    def __init__(self,data_type):
        self.clips_path = 'data/covost/cv-corpus-15.0-2023-09-08/en/clips/'
        self.tsv_file_path = 'data/covost/cv-corpus-15.0-2023-09-08/en/covost_v2.en_zh-CN.' + data_type + '.tsv'
        self.audio_text_list =  pd.read_csv(self.tsv_file_path, sep="\t", header=0, encoding="utf-8",
                       escapechar="\\", quoting=csv.QUOTE_NONE, na_filter=False)
        self.prompts = ['This speech is saying that']

    
    def __len__(self):
        return len(self.audio_text_list)
    
    def __getitem__(self, idx):
        return  self.clips_path + self.audio_text_list['path'][idx], self.audio_text_list['sentence'][idx], self.prompts[idx%len(self.prompts)]


def compute_asr_metrics(preds, labels):
    assert len(preds) == len(labels)
    # return average word error rate
    return {"acc": (preds == labels).mean()}

    


def load_and_cache_asr_examples(data_type):
    return processors["asr"](data_type)


processors = {
    "asr": LibriSpeechASRProcessor,
}

output_modes = {
    "asr": "text"
}

