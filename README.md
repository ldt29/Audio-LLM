# Audio-LLM: Activating the Capabilities of Large Language Models to Comprehend Audio Data (ISNN 2024)
## setup
1. Our environment: The python version is 3.9.17, and other required packages can be installed with the following command: ```pip install -r requirements.txt```.
2. Download [whisper tiny](https://huggingface.co/openai/whisper-tiny/tree/main) to ```whisper_path```.
3. Download [Fine-tuned BEATs_iter3+ (AS2M) (cpt2)](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) to `beats_path`.
4. Download [vicuna 7B v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) to ```vicuna_path```.
5. Download [LibriSpeech](http://www.openslr.org/12/) to ```data```


## pre-train 
```
torchrun --master_port 23344 train_pre.py --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --output_dir output --do_train --do_eval --evaluate_during_training --overwrite_output_dir --local_rank 0
```
## eval
```
torchrun --master_port 23343 train_pre.py --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --do_eval --output_dir output --local_rank 0
```

## Inference
```
python inference.py  --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --ckpt_path output/best_model.pt
```

```
data/LibriSpeech/test-clean/237/134500/237-134500-0015.flac
What information can you get from this speech?
```

## web-demo
```
python web_demo.py  --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --ckpt_path output/best_model.pt
```

## ASR
wav path:
```
data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac
```
prompt:
```
please convert this speech to text.
Based on this speech, please write a storyline.
```
### train parallel 
```
python train_asr.py --train_data_dir data/LibriSpeech/train-clean-100 --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --output_dir output_asr --do_train --do_eval --evaluate_during_training --overwrite_output_dir
```
### torchrun one gpu
```
torchrun --master_port 23343 train_asr.py --train_data_dir data/LibriSpeech/train-clean-100 --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --output_dir output_asr --do_train --do_eval --evaluate_during_training --overwrite_output_dir --local_rank 0
```

### Inference
```
python inference.py  --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --ckpt_path output_asr/best_model.pt
```



## ER
wav path:
```
data/IEMOCAP/Session5/sentences/wav/Ses05F_impro01/Ses05F_impro01_F010.wav
```
prompt:
```
Please give me the emotion of this speech.
```
### train parallel 
```
python train_er.py --train_data_dir data --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --output_dir output_er --do_train --do_eval --evaluate_during_training --overwrite_output_dir
```
### torchrun one gpu
```
torchrun --master_port 23343 train_ft.py --train_data_dir data --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --output_dir output_er --do_train --do_eval --evaluate_during_training --overwrite_output_dir --local_rank 0 --ckpt_path output/best_model.pt
```


### Inference
```
python inference.py  --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --ckpt_path output_er/best_model.pt
```
