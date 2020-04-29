#!/usr/bin/env python3

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_distances
import copy


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

    df_q = pd.read_csv(args.questions, sep='\t', dtype=str)
    df_e = pd.DataFrame(explanations, columns=('uid', 'text'))
    
    answer_list = df_q['AnswerKey']
    question_list = copy.deepcopy(df_q['Question'])
    
    for i in range(0, len(question_list)):
        answer = answer_list[i]
        
        if question_list[i].find('(A)') != -1:
            start_A = question_list[i].index('(A)')
        if question_list[i].find('(B)') != -1:
            start_B = question_list[i].index('(B)')
        if question_list[i].find('(C)') != -1:
            start_C = question_list[i].index('(C)')
        if question_list[i].find('(D)') != -1:
            start_D = question_list[i].index('(D)')

        if question_list[i].find('(A)') != -1 and answer == 'A':
            content = question_list[i][start_A+4:start_B]
            question_list[i] = question_list[i][0:start_B]
            question_list[i] = question_list[i].replace('(A)', '')
        elif question_list[i].find('(B)') != -1 and answer == 'B':
            content = question_list[i][start_B+4:start_C]
            question_list[i] = question_list[i][0:start_A] + question_list[i][start_B:start_C]
            question_list[i] = question_list[i].replace('(B)', '')
        elif question_list[i].find('(C)') != -1 and answer == 'C':
            content = question_list[i][start_C+4:start_D]
            question_list[i] = question_list[i][0:start_A] + question_list[i][start_C:start_D]
            question_list[i] = question_list[i].replace('(C)', '')
        elif question_list[i].find('(D)') != -1 and answer == 'D':
            content = question_list[i][start_D+4:len(question_list[i])]
            question_list[i] = question_list[i][0:start_A] + question_list[i][start_D:len(question_list[i])]
            question_list[i] = question_list[i].replace('(D)', '')

        question_list[i] += content
        question_list[i] += content
        question_list[i] += content
        question_list[i] += content
        
    vectorizer = TfidfVectorizer().fit(question_list).fit(df_e['text'])
    #vectorizer = TfidfVectorizer().fit(question_list).fit(df_e['text'])
    #vectorizer = CountVectorizer().fit(question_list).fit(df_e['text'])
    X_q = vectorizer.transform(question_list)
    X_e = vectorizer.transform(df_e['text'])
    X_dist = cosine_distances(X_q, X_e)

    for i_question, distances in enumerate(X_dist):
        for i_explanation in np.argsort(distances)[:args.nearest]:
            print('{}\t{}'.format(df_q.loc[i_question]['questionID'], df_e.loc[i_explanation]['uid']))


if '__main__' == __name__:
    main()
