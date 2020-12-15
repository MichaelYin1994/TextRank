#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:20:47 2020

@author:     zhuoyin94
@reference:  https://github.com/someus/
@email:      zhuoyin94@163.com
@github:     https://github.com/MichaelYin1994
"""

import os
import numpy as np
import networkx as nx
from numba import njit, jit
np.random.seed(2020)


class AttrDict(dict):
    """Dict that can get attribute by dot"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def combine(word_list, window_size=2):
    if window_size < 2:
        window_size = 2

    for i in range(1, window_size):
        if i >= len(word_list):
            break
        word_list_tmp = word_list[i:]
        word_pair_zip = zip(word_list, word_list_tmp)
        for word_pair in word_pair_zip:
            yield word_pair


def compute_jaccard_similarity(word_list_0, word_list_1):
    vocab_list = list(set(word_list_0 + word_list_1))
    count_vec_0 = [word_list_0.count(word) for word in vocab_list]
    count_vec_1 = [word_list_1.count(word) for word in vocab_list]

    sim_vec_0 = [count_vec_0[i] * count_vec_1[i]  for i in range(len(vocab_list))]
    sim_vec_1 = [1 for item in sim_vec_0 if item > 0]
    n_words_cooccur = sum(sim_vec_1)

    if abs(n_words_cooccur) <= 0:
        return 0

    denominator = len(vocab_list)
    if abs(denominator) <= 0:
        return 0

    return n_words_cooccur / denominator


def compute_edit_similarity():
    pass


def compute_lcss_similarity(word_list_0=None, word_list_1=None, max_pos_diff=3):
    length_0, length_1 = len(word_list_0), len(word_list_1)
    norm_factor = min(length_0, length_1)

    if length_0 == 0 or length_0 == 0:
        return 0
    if length_0 == 1:
        if word_list_0[0] in word_list_1:
            return 1 / norm_factor
        else:
            return 0
    if word_list_1 == 1:
        if word_list_1[0] in word_list_0:
            return 1 / norm_factor
        else:
            return 0

    dp = np.zeros((length_0 + 1, length_1 + 1))
    for i in range(1, length_0 + 1):
        for j in range(1, length_1 + 1):
            pos_diff = abs(i - j)

            if (pos_diff <= max_pos_diff) and (word_list_0[i-1] == word_list_1[j-1]):
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])
    return dp[-1, -1] / norm_factor

# TODO: Important parameters explain
# vertex_source, edge_source, _vertex_source, _edge_source, pagerank_config
def sort_words(vertex_source, edge_source, window_size=2, pagerank_config=None):
    sorted_words = []
    word2index, index2word = {}, {}
    _vertex_source = vertex_source
    _edge_source = edge_source

    # Get the vertex set
    word_index = 0
    for word_list in _vertex_source:
        for word in word_list:
            if not word in word2index:
                word2index[word] = word_index
                index2word[word_index] = word
                word_index += 1

    # Construct the Boolean Adjacent Matrix
    adjacent_mat = np.zeros((word_index, word_index))
    for word_list in _edge_source:
        for word_0, word_1 in combine(word_list, window_size):
            if word_0 in word2index and word_1 in word2index:
                index_0 = word2index[word_0]
                index_1 = word2index[word_1]

                adjacent_mat[index_0][index_1] = 1
                adjacent_mat[index_1][index_0] = 1

    # PageRank for computing the importances of words
    graph = nx.from_numpy_matrix(adjacent_mat)
    vertex_scores = nx.pagerank(graph, **pagerank_config)

    sorted_scores = sorted(
        vertex_scores.items(), key=lambda item: item[1], reverse=True)
    for ind, score in sorted_scores:
        item = AttrDict(word=index2word[ind], weight=score)
        sorted_words.append(item)

    return sorted_words



def sort_sentences():
    pass
