# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import itertools
import csv
import fire

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tokenization
import models
import optim
import train
import pdb
import numpy as np
import pandas as pd

from utils import set_seeds, get_device, truncate_tokens_pair
import os

def read_explanations(path):
    header = []
    uid = None

    df = pd.read_csv(path, sep='\t', dtype=str)

    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        else:
            header.append(name)

    if not uid or len(df) == 0:
        print('Possibly misformatted file: ' + path)
        return []

    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isna(s))), 1).tolist()

tables = '/data/jacob/code/nlp/tfidf/data/annotation/expl-tablestore-export-2017-08-25-230344/tables'
questions = '/data/jacob/code/nlp/tfidf/data/questions/ARC-Elementary+EXPL-Dev.tsv'

def parse_e(e):
    l = e.split(' ')
    l = [ll.split('|')[0] for ll in l]
    return l
class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None
    def __init__(self, pipeline=[]): # cvs file and pipeline object
        Dataset.__init__(self)

        explanations = []

        for path, _, files in os.walk(tables):
            for file in files:
                explanations += read_explanations(os.path.join(path, file))

        if not explanations:
            warnings.warn('Empty explanations')

        df_q = pd.read_csv(questions, sep='\t', dtype=str)
        df_e = pd.DataFrame(explanations, columns=('uid', 'text'))

        # pdb.set_trace()
        q_list = []
        e_list = []
        dict_e = {}

        num_e = len(df_e['uid'])
        num_q = len(df_q['questionID'])

        for i in range(num_e):
            dict_e[df_e['uid'][i]]= df_e['text'][i]


        for i in range(num_q):
            if not df_q['explanation'][i] is np.nan:
                q_list.append(df_q['Question'][i])
                e_list.append(parse_e(df_q['explanation'][i]))

        self.q_list = q_list
        self.e_list = e_list
        self.dict_e = dict_e

        self.pipeline = pipeline

        self.es = list(dict_e.keys())

        self.num_neg = 75
        # pdb.set_trace()
        
        # data = []
        # with open(file, "r") as f:
        #     # list of splitted lines : line is also list
        #     lines = csv.reader(f, delimiter='\t', quotechar=None)
        #     pdb.set_trace()
        #     for instance in self.get_instances(lines): # instance : tuple of fields
        #         for proc in pipeline: # a bunch of pre-processing
        #             instance = proc(instance)
        #         data.append(instance)

        # # To Tensors
        # self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return len(self.q_list)

    def __getitem__(self, index):
        # pdb.set_trace()
        q = self.q_list[index]
        e = self.e_list[index]

        pos = self.dict_e[np.random.choice(e)]

        # neg = []
        samples = []
        instance = ('1', q, pos)
        for proc in self.pipeline:
            instance = proc(instance)
        samples.append(instance)

        for i in range(self.num_neg):
            # pdb.set_trace()
            neg = self.dict_e[np.random.choice(self.es)]
            instance = ('0', q, neg)
            for proc in self.pipeline:
                instance = proc(instance)
            samples.append(instance)

        # pdb.set_trace()
        data = [torch.tensor(x, dtype=torch.long) for x in zip(*samples)]
        # data = [d for d in zip(data)]

        return data




class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor # e.g. text normalization
        self.tokenize = tokenize # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) \
                   if text_b else []

        return (label, tokens_a, tokens_b)


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        input_mask = [1]*(len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, label_id)


class Classifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        # only use the first h in the sequence
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        logits = torch.exp(logits).clamp(0, 100)
        return logits

#pretrain_file='../uncased_L-12_H-768_A-12/bert_model.ckpt',
#pretrain_file='../exp/bert/pretrain_100k/model_epoch_3_steps_9732.pt',
def neg_logloss(logits):
    score = logits[0] / logits.sum()
    loss = -torch.log(score+1e-4)
    return loss

def main(task='mrpc',
         train_cfg='config/train_mrpc.json',
         model_cfg='config/bert_base.json',
         data_file='../glue/MRPC/train.tsv',
         model_file=None,
         pretrain_file='../uncased_L-12_H-768_A-12/bert_model.ckpt',
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='../exp/bert/mrpc',
         max_len=128,
         mode='train'):

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)

    pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                AddSpecialTokensWithTruncation(max_len),
                TokenIndexing(tokenizer.convert_tokens_to_ids,
                              ('0', '1'), max_len)]
    dataset = CsvDataset(pipeline)
    # print(dataset[0])
    # pdb.set_trace()
    data_iter = DataLoader(dataset, batch_size=1, shuffle=True)

    model = Classifier(model_cfg, 1)
    criterion = nn.CrossEntropyLoss()

    trainer = train.Trainer(cfg,
                            model,
                            data_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, get_device())

    if mode == 'train':
        def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
            # pdb.set_trace()
            input_ids, segment_ids, input_mask, label_id = [b[0] for b in batch]
            # pdb.set_trace()
            logits = model(input_ids, segment_ids, input_mask)
            # pdb.set_trace()
            loss = neg_logloss(logits)
            # loss = criterion(logits, label_id)
            return loss

        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    elif mode == 'eval':
        def evaluate(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float() #.cpu().numpy()
            accuracy = result.mean()
            return accuracy, result

        results = trainer.eval(evaluate, model_file, data_parallel)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy:', total_accuracy)


if __name__ == '__main__':
    fire.Fire(main)
