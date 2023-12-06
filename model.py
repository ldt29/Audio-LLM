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
from beats.BEATs import BEATsConfig, BEATs
from qformer.Qformer import BertConfig, BertLMHeadModel

class ALLM(nn.Module):
    def __init__(
        self,
        args,
        speech_qformer_token_num=1,
        speech_qformer_layer=2,
        lora=True,
        lora_alpha=32,
        lora_rank=8,
        lora_dropout=0.1,
        second_per_frame=0.333333,
        second_stride=0.333333,
        low_resource=False
    ):

        super().__init__()

        # feature_extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper_path)

        # whisper
        self.speech_encoder = WhisperModel.from_pretrained(args.whisper_path).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)

        # beats
        self.beats_ckpt = args.beats_path
        beats_checkpoint = torch.load(self.beats_ckpt, map_location='cpu')
        beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
        beats = BEATs(beats_cfg)
        beats.load_state_dict(beats_checkpoint['model'])
        self.beats = beats
        self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
        for name, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()

        # init speech Qformer
        self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            speech_qformer_token_num,
            self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim,
            speech_qformer_layer,
        )
        self.second_per_frame = second_per_frame
        self.second_stride = second_stride
        
        # vicuna
        if not low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.vicuna_path,
                torch_dtype=torch.float16,
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.vicuna_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': 0}
            )

        # lora
        self.lora = lora
        if lora:
            target_modules = None
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)

        # tokenizer
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.vicuna_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        self.llama_tokenizer.padding_side = "right"

        # proj
        self.speech_llama_proj = nn.Linear(
            self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size)

        # load ckpt
        if args.ckpt_path is not None:
            ckpt_dict = torch.load(args.ckpt_path)['model']
            self.load_state_dict(ckpt_dict, strict=False)

        self.loss_fc = nn.CrossEntropyLoss(ignore_index=self.llama_tokenizer.pad_token_id)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        inputs,
        prompt,
        labels,
        prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        device='cuda:0',
    ):
        # read wav
        wav_np = np.zeros((len(inputs), 30*16000))
        batch_size = 0
        for wav_path in inputs:
            wav, sr = sf.read(wav_path)
            if len(wav.shape) == 2:
                wav = wav[:, 0]
            if len(wav) > 30 * sr:
                wav = wav[: 30 * sr]
            if len(wav) < 30 * sr:
                wav = np.concatenate((wav, np.zeros(30 * 16000 - len(wav))))
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
            wav_np[batch_size] = wav
            batch_size += 1
        
        # whisper
        spectrogram = self.feature_extractor(wav_np, return_tensors="pt", sampling_rate=16000).input_features.to(device) # [1, 80, 3000]
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
       
        # beats
        raw_wav = torch.from_numpy(wav_np).to(device)
        audio_padding_mask = torch.zeros(raw_wav.shape, device=device).bool()
        audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)

        # auditory embeds
        speech_embeds = self.ln_speech(speech_embeds)
        audio_embeds = self.ln_audio(audio_embeds)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

        # split frames
        B, T, C = speech_embeds.shape
        kernel = round(T * self.second_per_frame / 30.0)
        stride = round(T * self.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        # Qformer
        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        prompt_left, prompts_right = prompt_pattern.format(prompt).split('<SpeechHere>')
        prompt_left_ids = self.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)
        prompt_left_embeds = prompt_left_embeds.repeat(batch_size,1,1)
        prompt_right_ids = self.llama_tokenizer(
            prompts_right,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids)
        prompt_right_embeds = prompt_right_embeds.repeat(batch_size,1,1)
        bos_embeds = self.llama_model.model.embed_tokens(
            torch.ones(
                [batch_size, 1],
                dtype=torch.long,
                device=device,
            ) * self.llama_tokenizer.bos_token_id
        ) if not self.lora else self.llama_model.model.model.embed_tokens(
            torch.ones(
                [batch_size, 1],
                dtype=torch.long,
                device=device,
            ) * self.llama_tokenizer.bos_token_id
        )
        # print(bos_embeds.shape) torch.Size([8, 1, 4096]) 
        # print(prompt_left_embeds.shape) torch.Size([8, 7, 4096])
        # print(speech_embeds.shape) torch.Size([8, 88, 4096])
        # print(prompt_right_embeds.shape) torch.Size([8, 16, 4096])
        labels_ids = self.llama_tokenizer(
            labels,
            return_tensors="pt",
            add_special_tokens=False,
            padding = True
        ).to(device).input_ids
        labels_embeds = embed_tokens(labels_ids)
        embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds, labels_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(device)
        embeds = embeds.to(torch.float16)

        # predict
        # print(embeds.shape)  torch.Size([8, 112, 4096])
        # print(labels_ids.shape) torch.Size([8, 125])
        outputs = self.llama_model(
            inputs_embeds=embeds,
            attention_mask=atts,
        )
        logits = outputs.logits[:,-labels_ids.size(1)-1:-1,:]
        
        logits = self.softmax(logits).transpose(1, 2)
        loss = self.loss_fc(logits, labels_ids)


        return loss,logits,labels_ids
        
    def generate(
        self,
        wav_path,
        prompt,
        prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        device='cuda:0',
        max_length=200,
        num_beams=4,
        do_sample=True,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
    ):
        # read wav
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
        
        # whisper
        spectrogram = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(device) # [1, 80, 3000]
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
       
        # beats
        raw_wav = torch.from_numpy(wav).to(device).unsqueeze(0)
        audio_padding_mask = torch.zeros(raw_wav.shape, device=device).bool()
        audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)

        # auditory embeds
        speech_embeds = self.ln_speech(speech_embeds)
        audio_embeds = self.ln_audio(audio_embeds)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

        # split frames
        B, T, C = speech_embeds.shape
        kernel = round(T * self.second_per_frame / 30.0)
        stride = round(T * self.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        # Qformer
        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        prompt_left, prompts_right = prompt_pattern.format(prompt).split('<SpeechHere>')
        prompt_left_ids = self.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)
        prompt_right_ids = self.llama_tokenizer(
            prompts_right,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids)

        bos_embeds = self.llama_model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=device,
            ) * self.llama_tokenizer.bos_token_id
        ) if not self.lora else self.llama_model.model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=device,
            ) * self.llama_tokenizer.bos_token_id
        )

        embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        # generate
        output = self.llama_model.generate(
            inputs_embeds=embeds,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            attention_mask=atts,
            bos_token_id=self.llama_tokenizer.bos_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.pad_token_id
        )
        
        output_text = self.llama_tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=True)

        return output_text

    def init_speech_Qformer(self, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
