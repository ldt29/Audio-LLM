## setup
1. Our environment: The python version is 3.9.17, and other required packages can be installed with the following command: ```pip install -r requirements.txt```.
2. Download [whisper tiny](https://huggingface.co/openai/whisper-tiny/tree/main) to ```whisper_path```.
3. Download [Fine-tuned BEATs_iter3+ (AS2M) (cpt2)](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) to `beats_path`.
4. Download [vicuna 7B v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) to ```vicuna_path```.
5. Download [salmonn v1](https://huggingface.co/tsinghua-ee/SALMONN/blob/main/salmonn_v1.pth) to ```ckpt_path```.
6. Download [LibriSpeech](http://www.openslr.org/12/) to ```data_dir```

### finetune
```
python train.py --train_data_dir data-asr/LibriSpeech/train-clean-100 --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --output_dir output --do_train --do_eval
```
### train 
```
python train.py --train_data_dir data-asr/LibriSpeech/train-clean-100 --vicuna_path lmsys/vicuna-7b-v1.5 --whisper_path openai/whisper-tiny --beats_path beats-path/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --output_dir output --do_train --do_eval
```
