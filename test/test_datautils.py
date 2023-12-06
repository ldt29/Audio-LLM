from data_utils import AudioProcessor
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


# test train_dataset
train_dataset = AudioProcessor("data-asr/LibriSpeech/test-clean")
audio_train_dataloader = DataLoader(
    train_dataset, batch_size=8
)
audio_epoch_iterator = tqdm(audio_train_dataloader, desc="Iteration")

print(train_dataset[1])