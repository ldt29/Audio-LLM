### finetune
```
python train.py --train_data_dir data-asr/LibriSpeech/train-clean-100 --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --output_dir output --do_train --do_eval
```
### train 
```
python train.py --train_data_dir data-asr/LibriSpeech/train-clean-100 --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --output_dir output --do_train --do_eval
```
