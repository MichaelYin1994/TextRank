#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 23:12:15 2020

@author:     zhuoyin94
@reference:  https://github.com/someus/
@email:      zhuoyin94@163.com
@github:     https://github.com/MichaelYin1994
"""

import networkx as nx
import numpy as np

from utils import compute_jaccard_similarity, sort_words, combine
from Segmentation import WordSegmentation


class TextRank4Keywords():
    def __init__(self, tokenizer=None):
        self.text = ''
        self.keywords = None
        self.sentences = None
        self.words_no_filter = None
        self.words_no_stop_words = None
        self.words_all_filters = None

        if tokenizer is None:
            self.tokenizer = WordSegmentation(is_lower=True,
                                              is_use_stop_words=False,
                                              is_use_word_tags_filter=False)
        else:
            self.tokenizer = tokenizer

    def analyze(self, text,
                window_size=2,
                vertex_source='all_filters',
                edge_source='no_stop_words',
                pagerank_config=None):
        pagerank_config = pagerank_config or {'alpha': 0.85}

        self.text = text
        self.word2index = {}
        self.index2word = {}
        self.keywords = []
        self.graph = None

        cutted_sentece_list = self.tokenizer.segment_sentence_list(text)
        self.keywords = sort_words(cutted_sentece_list,
                                   cutted_sentece_list,
                                   window_size=window_size,
                                   pagerank_config=pagerank_config)
        return self.keywords

        # result = self.seg.segment(text=text, lower=lower)
        # self.sentences = result.sentences
        # self.words_no_filter = result.words_no_filter
        # self.words_no_stop_words = result.words_no_stop_words
        # self.words_all_filters   = result.words_all_filters

        # util.debug(20*'*')
        # util.debug('self.sentences in TextRank4Keyword:\n', ' || '.join(self.sentences))
        # util.debug('self.words_no_filter in TextRank4Keyword:\n', self.words_no_filter)
        # util.debug('self.words_no_stop_words in TextRank4Keyword:\n', self.words_no_stop_words)
        # util.debug('self.words_all_filters in TextRank4Keyword:\n', self.words_all_filters)


        # options = ['no_filter', 'no_stop_words', 'all_filters']

        # if vertex_source in options:
        #     _vertex_source = result['words_'+vertex_source]
        # else:
        #     _vertex_source = result['words_all_filters']

        # if edge_source in options:
        #     _edge_source   = result['words_'+edge_source]
        # else:
        #     _edge_source   = result['words_no_stop_words']

        # self.keywords = util.sort_words(_vertex_source, _edge_source, window = window, pagerank_config = pagerank_config)
