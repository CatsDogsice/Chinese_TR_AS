import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.cuda import amp
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelWithLMHead,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed, get_linear_schedule_with_warmup, WEIGHTS_NAME,
)

from model import NERModel
from metricsutils import (
    compute_accuracy_labels, write_metrics,
    TOKEN_ACCURACY, SPAN_ACCURACY, MEAN_TOKEN_PRECISION,
    MEAN_TOKEN_RECALL, MEAN_SPAN_PRECISION, MEAN_SPAN_RECALL
)
from utils import ModelArguments, DataTrainingArguments
from utils import featureName2idx
from itertools import chain
from data_utils import load_and_cache_examples, get_labels
from model_utils import multi_source_label_refine, soft_frequency, mt_update, get_mt_loss, opt_grad
from eval import evaluate


MODEL_CLASSES = {
    "bert": (AutoConfig, NERModel, AutoTokenizer),
}

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def initialize(args, model_class, config, t_total, epoch):

    load_name_or_path = args.model_name_or_path \
        if args.lm_model_name_or_path is None \
        else args.lm_model_name_or_path



    model = NERModel.from_pretrained(
        load_name_or_path,
        from_tf=bool(".ckpt" in load_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)
    import torch, gc

    gc.collect()
    torch.cuda.empty_cache()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1,args.adam_beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if epoch == 0:
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    model.zero_grad()
    return model, optimizer, scheduler

def train(args, train_dataset, model_class, config, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    print("Train the model")
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'tfboard'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    #print()

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model, optimizer, scheduler = initialize(args, model_class, config, t_total, 0)
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


    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev, best_test = [0, 0, 0], [0, 0, 0]
    if args.mt:
        teacher_model = model
    self_training_teacher_model = model

    for epoch in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # Update labels periodically after certain begin step在某个开始步骤后定期更新标签
            if global_step >= args.self_training_begin_step:
                print("global_step = ", global_step, "开始self-train")

                # Update a new teacher periodically
                delta = global_step - args.self_training_begin_step
                if delta % args.self_training_period == 0:
                    self_training_teacher_model = copy.deepcopy(model)
                    self_training_teacher_model.eval()

                    # Re-initialize the student model once a new teacher is obtained
                    if args.self_training_reinit:
                        model, optimizer, scheduler = initialize(args, model_class, config, t_total, epoch)


                # Using current teacher to update the label
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                with torch.no_grad():
                    outputs = self_training_teacher_model(**inputs)

                label_mask = None
                if args.self_training_label_mode == "hard":  # 默认hard
                    pred_labels = torch.argmax(outputs[0], axis=2)
                    pred_labels, label_mask = multi_source_label_refine(args, batch[5], batch[3], pred_labels,
                                                                        pad_token_label_id, pred_logits=outputs[0])
                elif args.self_training_label_mode == "soft":
                    pred_labels = soft_frequency(logits=outputs[0], power=2)
                    pred_labels, label_mask = multi_source_label_refine(args, batch[5], batch[3], pred_labels,
                                                                        pad_token_label_id)


                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": pred_labels,
                          "label_mask": label_mask}
            else:#
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )

            outputs = model(**inputs)
            # loss, logits, final_embeds = outputs[0], outputs[1], outputs[2]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss, logits = outputs[0], outputs[1]
            mt_loss, vat_loss = 0, 0


            loss = loss + args.mt_beta * mt_loss + args.vat_beta * vat_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:

                        logger.info("***** Entropy loss: %.4f, mean teacher loss : %.4f; vat loss: %.4f *****", \
                                    loss - args.mt_beta * mt_loss - args.vat_beta * vat_loss, \
                                    args.mt_beta * mt_loss, args.vat_beta * vat_loss)

                        results, _, best_dev, _ ,_= evaluate(args, model, tokenizer, labels, pad_token_label_id, best_dev,
                                                           mode="dev",
                                                           prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step,
                                                                                                          t_total,
                                                                                                          epoch,
                                                                                                          args.num_train_epochs),
                                                           verbose=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        results, _, best_test, is_updated, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id,
                                                                     best_test, mode="test",
                                                                     prefix='test [Step {}/{} | Epoch {}/{}]'.format(
                                                                         global_step, t_total, epoch,
                                                                         args.num_train_epochs), verbose=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("test_{}".format(key), value, global_step)

                        output_dirs = []
                        if args.local_rank in [-1, 0] and is_updated:
                            updated_self_training_teacher = True
                            output_dirs.append(os.path.join(args.output_dir, "checkpoint-best"))

                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                            output_dirs.append(os.path.join(args.output_dir, "checkpoint-{}".format(global_step)))

                        if len(output_dirs) > 0:
                            for output_dir in output_dirs:
                                logger.info("Saving model checkpoint to %s", args.output_dir)
                                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                                # They can then be reloaded using `from_pretrained()`
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)
                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model, global_step, tr_loss / global_step, best_dev, best_test
#luo
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='/root/data/MSRA',
        type=str,
        required=False,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=False,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='/root/bert_chinese_base',
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: "
    )#/root/autodl-tmp/NEEDLE/bio_script/roberta-base

    parser.add_argument(
        "--lm_model_name_or_path",
        default='/root/bert_chinese_base',#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        type=str,
        required=False,
        help="Path to lm model fine-tuned on fba data "
    )
    parser.add_argument(
        "--output_dir",
        default='/root/data/MSRA/out',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="/root/bert_chinese_base", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="/root/bert_chinese_base",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    #parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.")

    parser.add_argument("--do_eval", type=bool, default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_noise_eval", type=bool, default=False, help="Whether to run eval on the dev set.")

    parser.add_argument("--do_predict", type=bool, default=True, help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        type=bool,default=True,
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=24, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=24, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="BETA1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="BETA2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1, type=float, help="Total number of training epochs to perform."
    )#***********************
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=20, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=168, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=168, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        type=bool, default=False,
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",type=bool, default=True, help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    # mean teacher
    parser.add_argument('--mt', type=int, default=0, help='mean teacher.')
    parser.add_argument('--mt_updatefreq', type=int, default=1, help='mean teacher update frequency')
    parser.add_argument('--mt_class', type=str, default="kl",
                        help='mean teacher class, choices:[smart, prob, logit, kl(default), distill].')
    parser.add_argument('--mt_lambda', type=float, default=1, help="trade off parameter of the consistent loss.")
    parser.add_argument('--mt_rampup', type=int, default=300, help="rampup iteration.")
    parser.add_argument('--mt_alpha1', default=0.99, type=float,
                        help="moving average parameter of mean teacher (for the exponential moving average).")
    parser.add_argument('--mt_alpha2', default=0.995, type=float,
                        help="moving average parameter of mean teacher (for the exponential moving average).")
    parser.add_argument('--mt_beta', default=20, type=float, help="coefficient of mt_loss term.")
    parser.add_argument('--mt_avg', default="exponential", type=str,
                        help="moving average method, choices:[exponentail(default), simple, double_ema].")
    parser.add_argument('--mt_loss_type', default="logits", type=str,
                        help="subject to measure model difference, choices:[embeds, logits(default)].")

    # virtual adversarial training
    parser.add_argument('--vat', type=int, default=0, help='virtual adversarial training.')
    parser.add_argument('--vat_eps', type=float, default=1e-3,
                        help='perturbation size for virtual adversarial training.')
    parser.add_argument('--vat_lambda', type=float, default=1,
                        help='trade off parameter for virtual adversarial training.')
    parser.add_argument('--vat_beta', type=float, default=1,
                        help='coefficient of the virtual adversarial training loss term.')
    parser.add_argument('--vat_loss_type', default="logits", type=str,
                        help="subject to measure model difference, choices = [embeds, logits(default)].")

    # self-training
    parser.add_argument('--self_training_reinit', type=int, default=0,
                        help='re-initialize the student model if the teacher model is updated.')
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    parser.add_argument('--self_training_begin_step', type=int, default=10000,
                        help='the begin step (usually after the first epoch) to start self-training.')
    parser.add_argument('--self_training_label_mode', type=str, default="soft",
                        help='pseudo label type. choices:[hard(default), soft].')
    parser.add_argument('--self_training_period', type=int, default=5000, help='the self-training period.')
    parser.add_argument('--self_training_hp_label', type=float, default=7.9, help='use high precision label.')
    parser.add_argument('--self_training_ensemble_label', type=int, default=0, help='use ensemble label.')

    # Use data from weak.json************************
    parser.add_argument('--load_weak', type=bool, default=False, help='Load data from weak.json.')

    parser.add_argument('--remove_labels_from_weak', type=bool, default=True,
                        help='Use data from weak.json, and remove their labels for semi-supervised learning')
    parser.add_argument('--rep_train_against_weak', type=int, default=1,
                        help='Upsampling training data again weak data. Default: 1')
  #新增
    parser.add_argument('--loss_func',type=str, default="CrossEntropyLoss", help= "loss function for token classifer"
    )
    parser.add_argument('--crf_loss_func', type=str, default="nll", help='loss function'
                        )
    parser.add_argument('--feature_names', type=str, default=None, help='list of feature names, separated by ,'
                        )



    args = parser.parse_args()



    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    # Setup distant debugging if needed #
    if args.server_ip and args.server_port:

        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    nerProcessor = DataProcessor(args.data_dir)
    label_list = nerProcessor.get_labels()
    labels = get_labels(args.data_dir)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
   # pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    config = AutoConfig.from_pretrained(
        args.model_name_or_path
        if args.config_name is None
        else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir = args.cache_dir if args.cache_dir else None,
    )
    # always overwrite loss function
    config.loss_func = args.loss_func
    features_dim = nerProcessor.get_features_dim()
    if not hasattr(config, 'features_dict'):
        config.features_dict = {}
        if args.feature_names is not None:
            feature_names = args.feature_names.split(',')
            feature_list = [int(featureName2idx[feature]) for feature in feature_names]
            for feature_idx in feature_list:
                config.features_dict[feature_idx] = args.feature_dim
    if not hasattr(config, 'features_dim'):
        config.features_dim = features_dim

    if not hasattr(config, 'use_cnn'):
        config.use_cnn = args.use_cnn
    if not hasattr(config, 'cnn_kernels'):
        config.cnn_kernels = args.cnn_kernels
    if not hasattr(config, 'cnn_out_channels'):
        config.cnn_out_channels = args.cnn_out_channels

    if not hasattr(config, 'use_crf'):
        config.use_crf = args.use_crf
    if args.weak_wei_file is not None:
        assert config.use_crf, "not implemented for non crf model"
    if config.use_crf:
        config.loss_func = args.crf_loss_func
    features_dim = nerProcessor.get_features_dim()
    label_list = nerProcessor.get_labels()
    label_map = nerProcessor.get_label_map()
    inversed_label_map = nerProcessor.get_invsered_label_map()
    config.inversed_label_map = inversed_label_map
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path
        if args.tokenizer_name is None
        else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,
        config=config,
    )


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        label_map['[CLS]'] = pad_token_label_id
        label_map['[SEP]'] = pad_token_label_id
        label_map['X'] = pad_token_label_id

        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        if args.load_weak:
            weak_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="weak",
                                                   remove_labels=args.remove_labels_from_weak)
            train_dataset = torch.utils.data.ConcatDataset(
                [train_dataset] * args.rep_train_against_weak + [weak_dataset, ])

        model, global_step, tr_loss, best_dev, best_test = train(args, train_dataset, NERModel, config, tokenizer,
                                                                 labels, pad_token_label_id)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving last-practice: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        if not best_dev:
            best_dev = [0, 0, 0]
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = NERModel.from_pretrained(checkpoint)
            model.to(args.device)
            result, predictions, best_dev, _ ,_= evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_dev,
                                              mode="dev", prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))


    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = NERModel.from_pretrained(args.output_dir)
        model.to(args.device)
#
        if not best_test:
            best_test = [0, 0, 0]
        result, predictions, _,_, high = evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_test,
                                             mode="test")
        # Save results

        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "test.json"), "r") as f:
                example_id = 0
                data = json.load(f)
                sample = {}
                texts = []
                for idx in high:
                # for item in data:
                    sample['text'] = data[idx]["text"]
                    sample['label'] = predictions[idx]
                    texts.append(sample)

                    sample = {}
                    example_id += 1
            json_data = json.dumps(texts, ensure_ascii=False)
            writer.write("{}\n".format(json_data))

    return results
if __name__ == '__main__':
    main()