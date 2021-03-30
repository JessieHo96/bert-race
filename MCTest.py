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
"""BERT finetuning runner."""

from __future__ import absolute_import

import argparse
import csv
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch

import glob
import json

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForMultipleChoice, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np


# remove redundancy word and make it can be open by pandas
def modify_data(file):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.replace(';Work', '\tWork')
            line = line.replace('Author: ', '')
            line = line.replace('Work Time(s): ', '')
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


# get all answer of the question
def get_answer(file):
    output_list = pd.read_csv(file, sep='	', header=None)
    return output_list

# separate each 4 question to 4 columns
def separate_question(input_list):
    output_list = []
    if input_list[0].find('one:') != -1:
        output_list.append(input_list[0][0:3])
        output_list.append(input_list[0][5:])
    elif input_list[0].find('multiple:') != -1:
        output_list.append(input_list[0][0:8])
        output_list.append(input_list[0][10:])
    # print(input_list[1:])
    output_list.extend(input_list[1:])
    return output_list


def transfer(data_file, answer_file):
    modify_data(data_file)
    data = pd.read_csv(data_file, sep='\t', header=None)
    answer_data = get_answer(answer_file)
    df_new = pd.DataFrame()
    for index, row in data.iterrows():
        new_columns = row[0:4].values
        for i in range(4):
            # print(row[4 + 5 * i: 4 + 5 * i + 5].values)
            temp_list = separate_question(row[4 + 5 * i: 4 + 5 * i + 5].values)
            final_columns = np.append(new_columns, temp_list)
            final_columns = np.append(final_columns, answer_data[i][index])
            final_columns = final_columns.tolist()
            df_new = df_new.append([final_columns], ignore_index=True)
    df_new.columns = ['id', 'author', 'work_times', 'article', 'question_type', 'question', 'A', 'B', 'C', 'D',
                      'answer']
    return df_new




class MCTest(object):
    """We are going to train race dataset with bert."""
    def __init__(self,
                 id,
                 article,
                 question,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 answer = None):
        self.id = id
        self.question = question
        self.article = article
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.answer = answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.id),
            "article_sentence: {}".format(self.article),
            "question: {}".format(self.question),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
        ]

        if self.answer is not None:
            l.append("answer: {}".format(self.answer))

        return ", ".join(l)

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


# data_grade is a list of grades of the dataset.
# for example: ["high","middle"]
def read_MCtest(df):
    samples = []
    for item in df.iterrows():
        id = item[1][0]
        article = item[1][3]
        question = item[1][5]
        ending_0 = item[1][6]
        ending_1 = item[1][7]
        ending_2 = item[1][8]
        ending_3 = item[1][9]
        answer = item[1][10]
        # print(id, article, question, ending_0, ending_1, ending_2, ending_3, answer)

        # break

        samples.append(MCTest(id, article, question, ending_0, ending_1, ending_2, ending_3, answer))


    return samples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task like Swag. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Race example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    print(len(examples),examples[0])
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.article)
        start_ending_tokens = tokenizer.tokenize(example.question)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(start_ending_tokens)) +[2] * (len(tokenizer.tokenize(ending)) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.answer
        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("id: {}".format(example.id))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
            if is_training:
                logger.info("label: {}".format(label))
        if (example_index%5000 ==0): print(example_index)	
        features.append(
            InputFeatures(
                example_id = example.id,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    #print(outputs,outputs == labels)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    

    args = parser.parse_args()

    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
   
        
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
   

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    #print(args.output_dir)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        
        df_train = transfer(data_file=os.path.join(args.data_dir,'mc500.train.tsv'), answer_file=os.path.join(args.data_dir,'/mc500.train.ans'))
        df_train['answer'] = df_train['answer'].apply(lambda x: ord(x)-65)
        
        train_examples = read_MCtest(df_train)
        
        print(len(train_examples), args.train_batch_size,args.gradient_accumulation_steps)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        

    # Prepare model
    model = BertForMultipleChoice.from_pretrained(args.bert_model,
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
        num_choices=4)
    
    if args.fp16:
        model.half()
        
    model.to(device)
    

    #Freeze the embedding network and encoder layers
    print("Freeze network")
    for name, param in model.named_parameters():
        ln = 24
        if name.startswith('bert.encoder'):
        	l = name.split('.')
        	ln = int(l[3])
      
        if name.startswith('bert.embeddings') or ln < 6:
        	print(name)  
        	param.requires_grad = False


    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    

    global_step = 0
    if args.do_train:

        if args.fp16:
        	try:
            		from apex.optimizers import FP16_Optimizer
            		from apex.optimizers import FusedAdam
        	except ImportError:
            		raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        	optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        	if args.loss_scale == 0:
            		optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        	else:
            		optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        	warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)
        else:
        	optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        
        
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()

        output_train_file = os.path.join(args.output_dir, "train_results.txt")        
        loss_writer = open(output_train_file, "w",1)

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):

            tr_loss = 0
            last_tr_loss = 0

            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
#                 if n_gpu > 1:
#                     loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step/num_train_optimization_steps,
                                                                                 args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if nb_tr_examples % 512 == 0:
                    loss_log = (tr_loss - last_tr_loss)*1.0/512
                    print(nb_tr_examples,loss_log)
                    loss_writer.write("%d %f \n" % (nb_tr_examples,loss_log))
                    last_tr_loss = tr_loss

    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForMultipleChoice.from_pretrained(args.output_dir, num_choices=4)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        model = BertForMultipleChoice.from_pretrained(args.output_dir, num_choices=4)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model.to(device)


    if args.do_eval:
        
        df_eval = transfer(data_file=os.path.join(args.data_dir,'mc500.dev.tsv'), answer_file=os.path.join(args.data_dir,'/mc500.dev.ans'))
        df_eval['answer'] = df_eval['answer'].apply(lambda x: ord(x)-65)
        
        eval_examples = read_MCtest(df_eval)
#         eval_examples = read_race(os.path.join(args.data_dir,"dev"))
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, args.max_seq_length, True)
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)

        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        tr_loss = 0
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        total_logits = []
        total_labels = []
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            
            total_logits.append(logits)
            total_labels.append(label_ids)
	
        total_logits = np.concatenate(total_logits)
        total_labels = np.concatenate(total_labels)

        np.save(args.output_dir+"/logits.npy",total_logits)
        np.save(args.output_dir+"/labels.npy",total_labels)

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
