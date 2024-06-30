# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import Counter
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from ner_metrics import SeqEntityScore
from data_utils import load_and_cache_examples, tag_to_id, get_chunks
# from flashtool import Logger
logger = logging.getLogger(__name__)

def evaluate(args, model, tokenizer, labels, pad_token_label_id, best, mode, prefix="", verbose=True):
    
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    metric = SeqEntityScore(labels, markup="bio")

    logger.info("***** Running evaluation %s *****", prefix)
    if verbose:
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    alllogits = None
    model.eval()



    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        # return y_pred, pred_probs
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        if preds is None:
            print('preds is None')
            preds = logits.detach().cpu().numpy()
            if alllogits is None:
                alllogits = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:

            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            alllogits = np.append(alllogits, logits.detach().cpu().numpy(), axis=0)
    Preds = preds
    alllogits = torch.tensor(alllogits)
    eval_loss = eval_loss / nb_eval_steps
    preds2 = F.softmax(alllogits, dim=-1).detach().cpu().numpy()

    MaxP = np.max(preds2, axis=2)
    preds = np.argmax(preds, axis=2)

    Maxlist = MaxP.tolist()


#############################################################################
    high = []
    for i in range(len(Maxlist)):
        # min_value = min(Maxlist[i])
        # if min_value > 0.8:
            high.append(i)
    # for l in  Maxlist:
    #
    #     allo = all(element == 0 for element in l)
    #     if allo:
    #
    #         scoer.append(0)
    #     else:
    #         min_num = min(filter(lambda x: x > 0, l))
    #         scoer.append(min_num)



    label_map = {i: label for i, label in enumerate(labels)}
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    out_id_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_id_list = [[] for _ in range(out_label_ids.shape[0])]




    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):

            if out_label_ids[i, j] != pad_token_label_id:
                preds_list[i].append(label_map[preds[i][j]])
                #print(label_map[preds[i][j]])
                out_label_list.append(label_map[out_label_ids[i][j]])
                out_id_list[i].append(out_label_ids[i][j])
                preds_id_list[i].append(preds[i][j])

    for ground_truth_id, predicted_id in zip(out_id_list, preds_id_list):



        metric.update(pred_paths=[predicted_id], label_paths=[ground_truth_id])

    eval_info, entity_info = metric.result()

    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    new_F = results.get('f1')
    is_updated = False
    if new_F > best[-1]:
        best = [results.get('acc'), results.get('recall'), new_F]
        is_updated = True
    logger.info("***** Entity results %s*****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    # return results




    Results = {
       "loss": eval_loss,
       "precision": results.get('acc'),
       "recall": results.get('recall'),
       "f1": new_F,
       "best_precision": best[0],
       "best_recall":best[1],
       "best_f1": best[-1]
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return Results, preds_list, best, is_updated, high

