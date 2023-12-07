import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    LlamaForCausalLM,
    LlamaTokenizer
)
import numpy as np
import librosa


lora=True
lora_alpha=32
lora_rank=8
lora_dropout=0.1
second_per_frame=0.333333
second_stride=0.333333
low_resource=False
vicuna_path = "lmsys/vicuna-7b-v1.5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_path, use_fast=False)
llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

llama_tokenizer.pad_token_id = 0
llama_tokenizer.padding_side = "right"

llama_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=torch.float16,
            ).to(device)

target_modules = None
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=lora_rank, 
    lora_alpha=lora_alpha, 
    lora_dropout=lora_dropout,
    target_modules=target_modules,
)
llama_model = get_peft_model(llama_model, peft_config)

# tokenizer


labels = ["i like you very much.","i love you."]

labels_ids = llama_tokenizer(
            labels,
            return_tensors="pt",
            add_special_tokens=False,
            padding = True
        ).to(device).input_ids

embeds = llama_model.model.model.embed_tokens(
            labels_ids
        )
atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
print(embeds)

outputs = llama_model(
            inputs_embeds=embeds,
            labels = labels_ids,
            attention_mask=atts,
        )

print(outputs.loss)
print(outputs.logits)
preds = outputs.logits.detach().cpu().numpy()
preds = np.argmax(preds, axis=2)   
print(preds)
print(labels_ids)
print(llama_model.config.pad_token_id)