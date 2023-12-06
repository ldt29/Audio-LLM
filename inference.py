import torch
import argparse
from model import ALLM
       
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--whisper_path", type=str, default=None)
    parser.add_argument("--beats_path", type=str, default=None)
    parser.add_argument("--vicuna_path", type=str, default=None)
    parser.add_argument("--low_resource", action='store_true', default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    model = ALLM(
        args=args
    )
    model.to(args.device)
    model.eval()
    while True:
        print("=====================================")
        wav_path = input("Your Wav Path:\n")
        prompt = input("Your Prompt:\n")
        try:
            print("Output:")
            # for environment with cuda>=117
            with torch.cuda.amp.autocast(dtype=torch.float16):
                print(model.generate(wav_path, prompt=prompt)[0])
        except Exception as e:
            print(e)
            if args.debug:
                import pdb; pdb.set_trace()
