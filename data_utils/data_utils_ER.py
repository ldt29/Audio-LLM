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

logger = logging.getLogger(__name__)

def get_txt_files(file_dir):
        L = []
        for root, dirs, files, in os.walk(file_dir):
                for file in files:
                    if os.path.splitext(file)[1] == '.txt':
                        filename = os.path.join(root, file)
                        L.append(filename)
        return L

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
        print("error")

    return target



class ERProcessor(Dataset):
    def __init__(self, data_dir):

        assert os.path.isfile(data_dir)

        # read IEMOCAP file
        all_data = []
        Sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        target_dir = "/data/ER/"
        wav_dir = "/sentences/wav"
        
        for Session in Sessions:
            file_dir = data_dir + Session + target_dir
            txt_files = get_txt_files[file_dir]
            
            for txt_file in txt_files:
                last_folder = (txt_file.split("/")[-1]).split(".")[0]
                data = pd.read_csv(txt_file, delimiter="\n", skiprows=1, names='a')
                data['a'] = data['a'].astype(str)

                filter_data=[x for x in data['a'] if '[' in x]

                print(txt_file, len(filter_data))

                for file in filter_data:
                    values = file.split("\t")
                    filename = values[1] + ".wav"
                    filename = data_dir + Session + wav_dir + last_folder + "/" + filename

                    target = values[2]

                    target = get_IEMOCAP_9target(target)

                    if target == None:
                        print(target)
                        continue
                    result = (filename, target)
                    all_data.append(result)
        
        self.example = all_data
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    
def load_and_cache_examples(args,evaluate=False):
    pass




processors = {
    "wave2text": ERProcessor,
}

output_modes = {
    "wave2text": "text"
}


