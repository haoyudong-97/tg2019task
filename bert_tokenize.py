#!/usr/bin/env python3

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import random

#import tensorflow as tf
#import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
#import seaborn as sns

import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer

SEQ_LENGTH = 80
BATCH_SIZE = 20

def load_gold(filepath_or_buffer, sep='\t'):
    df = pd.read_csv(filepath_or_buffer, sep=sep, dtype=str)

    gold = OrderedDict()

    for _, row in df[['questionID', 'explanation']].dropna().iterrows():
        explanations = OrderedDict((uid.lower(), Explanation(uid.lower(), role))
                                   for e in row['explanation'].split()
                                   for uid, role in (e.split('|', 1),))

        question = Question(row['questionID'].lower(), explanations)

        gold[question.id] = question

    return gold

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
        warnings.warn('Possibly misformatted file: ' + path)
        return []

    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isna(s))), 1).tolist()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nearest', type=int, default=10)
    parser.add_argument('tables')
    parser.add_argument('questions', type=argparse.FileType('r', encoding='UTF-8'))
    args = parser.parse_args()

    explanations = []

    for path, _, files in os.walk(args.tables):
        for file in files:
            explanations += read_explanations(os.path.join(path, file))

    if not explanations:
        warnings.warn('Empty explanations')

    # prepare data
    df_q = pd.read_csv(args.questions, sep='\t', dtype=str)
    text_q = [q for q in df_q['Question']]
    df_e = pd.DataFrame(explanations, columns=('uid', 'text'))
    text_e = [e for e in df_e['text']]

    # answer id + text set
    set_e = {}
    for i in range(len(df_e)):
        set_e[df_e['uid'][i]] = df_e['text'][i]
    all_uids = list(set_e.keys()) 
    
    # prepare question + explanation pairs
    q_e_pairs = []
    q_e_pairs_label = []
    for i in range(len(text_q)):
        # find real question + explanations pairs
        question_text = text_q[i]
        try:
            explanations = df_q['explanation'][i].strip().split('|')[:-1]
        except:
            print('This question does not have explanation')
            continue
        clean_exp_id = [explanations[0]] + [x.split()[1] for x in explanations[1:]]
        explanation_texts = [set_e[e_id] for e_id in clean_exp_id]
        for explanation in explanation_texts:
            q_e = question_text + ' [SEP] ' + explanation
            q_e_pairs.append(q_e)
            q_e_pairs_label.append(1)
            
        # generate negative pairs by giving a random explanation to the question
        positive_number = len(explanation_texts)
        random_id = all_uids[random.randint(0, len(all_uids)-1)]
        count = 0
        while count != positive_number:
            if random_id not in clean_exp_id:
                random_explanation = set_e[random_id]
                false_q_e = question_text + ' [SEP] ' + random_explanation
                q_e_pairs.append(false_q_e)
                q_e_pairs_label.append(0)
                count += 1


    # build encoder
    #module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    #embed = hub.Module(module_url)

    ## get embedding
    #with tf.Session() as session:
    #    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #    question_embeddings = session.run(embed(text_q))
    #    answer_embeddings = session.run(embed(text_e))

    # define BERT to tokenize captions
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased').cuda()
    
    # train BERT with q_e pairs
    # map the hidden states of first token to label, 768 -> size of BERT
    fc = nn.Linear(768, 2).cuda()
    fc = torch.load('linear.pt')
    optimizer = torch.optim.Adam(fc.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    # train the fc layer
    #for i in range(200):
    #    count = 0
    #    input_batch = []
    #    label_batch = []
    #    # shuffle the data
    #    tmp = list(zip(q_e_pairs, q_e_pairs_label))
    #    random.shuffle(tmp)
    #    q_e_pairs[:], q_e_pairs_label[:] = zip(*tmp)
    #    for j in range(len(q_e_pairs)):
    #        q_e = q_e_pairs[j]
    #        label = q_e_pairs_label[j]
    #        input_token = tokenizer.encode(q_e)
    #        # fix padding
    #        if len(input_token) < SEQ_LENGTH:
    #            input_token += [0] * (SEQ_LENGTH - len(input_token))
    #        else:
    #            input_token = input_token[:SEQ_LENGTH]
    #        input_batch.append(input_token)
    #        label_batch.append(label)

    #        if j % BATCH_SIZE == 0 and j != 0:
    #            optimizer.zero_grad()
    #            input_batch = torch.tensor(input_batch).cuda()
    #            label_batch = torch.tensor(label_batch).cuda()
    #            outputs = bert(input_batch)
    #            first_cls_states = outputs[0][:,0,:]
    #            predict = fc(first_cls_states)
    #            output = loss(predict, label_batch)
    #            output.backward()
    #            optimizer.step()
    #            count += BATCH_SIZE
    #            if count % 300 == 0:
    #                print('epoch : ' + str(i) + ' iter: ' + str(count) + ' loss: ' + str(output.item()))
    #            input_batch = []
    #            label_batch = []


    #    torch.save(fc, 'linear.pt')
    #print('Finish Training')

    ##------------ end training -----------

    # predict question vs all explanations
    softmax = nn.Softmax(dim=1)
    total_prob = []
    for i, q in enumerate(text_q):
        print(i)
        explanation_prob = []
        input_batch = []
        for j, e in enumerate(text_e):
            if j % 500 == 0:
                print(j)
            input_pair = q + ' [SEP] ' + e
            input_token = tokenizer.encode(input_pair)
            # fix padding
            if len(input_token) < SEQ_LENGTH:
                input_token += [0] * (SEQ_LENGTH - len(input_token))
            else:
                input_token = input_token[:SEQ_LENGTH]
            input_batch.append(input_token)
           
            if j % BATCH_SIZE == 0 and j != 0:
                input_batch = torch.tensor(input_batch).cuda()
                output = bert(input_batch)
                first_cls_states = output[0][:,0,:]
                predict_prob = softmax(fc(first_cls_states))

                pos_label = predict_prob[:,1].tolist()
                explanation_prob += pos_label
                input_batch = []
        
        print(explanation_prob)
        break
        total_prob.append(explanation_prob)

    #for q_e in zip(q_e_pairs, q_e_pairs_label):
    #    token_q_e = tokenizer.encode(q_e)
    #    input_id = torch.tensor(token_q_e).unsqueeze(0)
    #    outputs = bert(input_id)
    #    first_cls_states = outputs[0][:,0,:]
    #    predict = fc(first_cls_states)
    #    predict_prob = softmax(predict)
    #    print(predict_prob)
    #    print(label)
    #    break


    ## get emebedding
    #question_embeddings = []
    #answer_embeddings = []
    #with torch.no_grad():
    #    for single_q in text_q:
    #        token_q = tokenizer.encode(single_q)
    #        input_id = torch.tensor(token_q).unsqueeze(0)
    #        last_hidden = bert(input_id)[0]
    #        last_hidden_mean = torch.mean(last_hidden, dim=1)[0]
    #        question_embeddings.append(last_hidden_mean.tolist())

    #    for single_e in text_e:
    #        token_e = tokenizer.encode(single_e)
    #        input_id = torch.tensor(token_e).unsqueeze(0)
    #        last_hidden = bert(input_id)[0]
    #        last_hidden_mean = torch.mean(last_hidden, dim=1)[0]
    #        answer_embeddings.append(last_hidden_mean.tolist())
    
    #X_q = vectorizer.transform(df_q['Question'])
    #X_e = vectorizer.transform(df_e['text'])
    #X_dist = cosine_distances(question_embeddings, answer_embeddings)

    #for i_question, distances in enumerate(X_dist):
    doc_str = ""
    for i_question, distances in enumerate(total_prob):
        for i_explanation in np.argsort(distances)[:args.nearest]:
            doc_str += '{}\t{}'.format(df_q.loc[i_question]['questionID'], df_e.loc[i_explanation]['uid'])
            doc_str += '\n'

    with open('output_total.txt', 'w+') as f:
        f.write(doc_str)

if '__main__' == __name__:
    main()
