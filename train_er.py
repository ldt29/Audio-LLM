import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from model import ALLM
from optimizer import AdamW, get_linear_schedule_with_warmup
from data_utils.data_utils_ER import compute_metrics, load_and_cache_examples  

from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    ) 
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.output_dir, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.output_dir, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.output_dir):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.output_dir.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    prev_tr_loss=tr_loss
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    best_dev_acc=0
    saved_model=False
    for _ in train_iterator:
        iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        #########################################  Your Code  ###########################################
        # todo
        # get batch and batch by iterating the text_poch_iterator and audio_epoch_iterator
        for step, batch in enumerate(iterator):
        
        #################################################################################################

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            ###################################################################################
            inputs = batch[0]
            labels = batch[1]
            prompt = batch[2][0]

            outputs = model(inputs,prompt, labels,device=args.device)
            # compute word error rate
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(iterator)
            ):

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if (
                        args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        dev_acc = evaluate(args, model, data_type="dev")['acc']
                        if dev_acc > best_dev_acc:
                            best_dev_acc = dev_acc
                            save_model(model, optimizer, scheduler, args, args.output_dir)
                            saved_model=True
                        print('loss this epoch',tr_loss-prev_tr_loss)
                        prev_tr_loss=tr_loss
            if args.max_steps > 0 and global_step > args.max_steps:
                iterator.close()
                break
        
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    logger.info(" best valid acc = {}".format(best_dev_acc))
    if saved_model==False:
        save_model(model, optimizer, scheduler, args, args.output_dir)
    return global_step, tr_loss / global_step


def save_model(model,optimizer,scheduler,args,output_dir):               
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = (
        model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
    save_info = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    filepath=os.path.join(args.output_dir, "best_model.pt")
    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def evaluate(args, model, data_type="test"):
    eval_output_dir=args.output_dir
    results = {}
    eval_dataset = load_and_cache_examples(args, data_type=data_type)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(data_type))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        inputs = batch[0]
        labels = batch[1]
        prompt = batch[2][0]

        with torch.no_grad():
            outputs = model(inputs, prompt, labels,device=args.device)
            tmp_eval_loss, logits, labels_ids = outputs[:3]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = np.argmax(logits.detach().cpu().numpy(), axis=2)  
            out_label_ids = labels_ids.detach().cpu().numpy()
        else:
            if len(preds) == len(logits): 
                preds = np.append(preds, np.argmax(logits.detach().cpu().numpy(), axis=2), axis=1)
                out_label_ids = np.append(out_label_ids, labels_ids.detach().cpu().numpy(), axis=1)

        eval_loss = eval_loss / nb_eval_steps
 

        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        if not os.path.exists(os.path.join(eval_output_dir, data_type)):
            os.makedirs(os.path.join(eval_output_dir, data_type))
        output_eval_file = os.path.join(eval_output_dir, data_type, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(data_type))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .wav files (or other data files) for the task.",
    )
    parser.add_argument(
        "--vicuna_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list:",
    )
    parser.add_argument(
        "--whisper_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list:",
    )
    parser.add_argument(
        "--beats_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list:",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list:",
    )
    parser.add_argument(
        "--dev_data_dir",
        default="data",
        type=str,
        help="The valid input data dir. Should contain the .wav files (or other data files) for the task.",
    )
    parser.add_argument(
        "--test_data_dir",
        default="data",
        type=str,
        help="The test input data dir. Should contain the .wav files (or other data files) for the task.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    # Set seed
    set_seed(args)


    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    model = ALLM(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        audio_train_dataset = load_and_cache_examples(args, data_type='train')
        global_step, tr_loss = train(args, audio_train_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # result = evaluate(args, model, tokenizer, prefix="")

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # tokenizer = BertTokenizer.from_pretrained(args.output_dir)
        saved = torch.load(os.path.join(args.output_dir,'best_model.pt'))
        model = ALLM(args)
        model.load_state_dict(saved['model'])
        model.to(args.device)
        result = evaluate(args, model, data_type='test')
    

if __name__ == "__main__":
    main()
