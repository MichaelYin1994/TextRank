#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:25:41 2020

@author:     zhuoyin94
@reference:  https://github.com/someus/
@email:      zhuoyin94@163.com
@github:     https://github.com/MichaelYin1994
"""

import os, re, warnings
from tqdm import tqdm
import numpy as np
import networkx as nx
import pkuseg
import pickle
np.random.seed(2020)
from Segmentation import WordSegmentation
from TextRank4Keywords import TextRank4Keywords

##############################################################################
def load_pkl(path=".//data//", filename=None):
    """Loading *.pkl from .//cached_data//"""
    with open(path+filename, 'rb') as file:
        data = pickle.load(file)
    return data


def load_stop_words():
    """Loading stop words vocab from the path .//cached_data//stopwords//"""
    path = ".//stopwords//"
    file_names = ["baidu_stopwords.txt", "cn_stopwords.txt",
                  "hit_stopwords.txt", "scu_stopwords.txt"]

    stop_words = []
    for name in file_names:
        with open(path+name, "r") as f:
            stop_words.extend(f.readlines())
    return stop_words


def load_test_corpus():
    path = ".//data//"
    file_names = [name for name in os.listdir(path) if ".txt" in name]

    corpus = []
    for name in file_names:
        with open(path+name, "r") as f:
            tmp = f.readlines()
            tmp = "".join(tmp)
            corpus.append(tmp)
    return corpus


if __name__ == "__main__":
    # Loadding data
    # --------------------------
    # corpus = load_pkl(
    #     filename="document_df.pkl")["fault_description"].values.tolist()
    corpus = load_test_corpus()
    user_vocab = load_pkl(filename="general_vocab.pkl") + load_pkl(filename="domain_vocab.pkl")
    user_vocab = list(set(user_vocab))
    stop_words = set(load_stop_words())

    # Testing of WordSegmentation
    # --------------------------
    # tokenizer = WordSegmentation(stop_words_vocab=stop_words,
    #                               user_vocab=user_vocab,
    #                               is_lower=True,
    #                               is_use_stop_words=False,
    #                               is_use_word_tags_filter=False)
    # corpus_cutted = tokenizer.segment_sentence_list(corpus)

    # Key word extraction
    # --------------------------
    tokenizer = WordSegmentation(stop_words_vocab=stop_words,
                                  user_vocab=user_vocab,
                                  is_lower=True,
                                  is_use_stop_words=False,
                                  is_use_word_tags_filter=True)
    textrank = TextRank4Keywords(tokenizer=tokenizer)

    res = []
    for text in corpus:
        res.append(textrank.analyze([text]))

    # seg = pkuseg.pkuseg(user_dict=vocab)
    # seg_postsg = pkuseg.pkuseg(postag=True)
    # stop_words = set(load_stop_words())
    # cop = re.compile(u"[^\u4e00-\u9faf^*^a-z^A-Z^0-9]")

    # # Cutting
    # sent_ind_0, sent_ind_1 = 121, 122
    # raw_sent_0 = corpus[sent_ind_0]
    # cutted_sent_0 = seg.cut(raw_sent_0)

    # raw_sent_1 = corpus[sent_ind_1]
    # cutted_sent_1 = seg.cut(raw_sent_1)

    # # Testing
    # print(compute_jaccard_similarity(cutted_sent_0, cutted_sent_1))
    # print(compute_lcss_similarity(cutted_sent_0, cutted_sent_1, max_pos_diff=2))
