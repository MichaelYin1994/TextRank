#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Thu Nov 26 23:12:15 2020
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994
# Reference:  https://github.com/someus/

"""
本模块(textrank.textrank4keywords)用于抽取具体文本中的关键词项。
"""

from segmentation import WordSegmentation
from utils import compute_word_scores

# TODO(zhuoyin94@163.com): 调用sklearn.utils的API对输入类型进行检测，完善异常处理
class TextRank4Keywords():
    """分词辅助。依据给定条件，将包含句子的列表切分为词的有序集合。

    @Parameters:
    ----------
        tokenizer: {bool-like}
            是否将文本中的英文字符置为小写，默认开启。

    @Attributes:
    ----------
        self.text:

        self.words_no_filter: {list-like}


    @Return:
    ----------
        分词或者分句之后的结果。代码实现参考了文献[1]。分词结果例如：
        [[”我“, "是", "中国", "公民"],
         ...
         ["北京", "工业", "大学"]]

    @References:
    ----------
    [1] https://github.com/letiantian/TextRank4ZH
    [2] https://github.com/lancopku/pkuseg-python
    """
    def __init__(self, tokenizer=None):
        self.text = ""
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

    def fit_predict(self, text,
                    window_size=2,
                    vertex_source="all_filters",
                    edge_source="no_stop_words",
                    pagerank_config=None):
        if not pagerank_config:
            pagerank_config = {"alpha": 0.85}

        # TODO(zhuoyin94@163.com): 此处进行了三次同样类型的分词，可提升效率
        # 不同种类的分词策略
        self.words_no_filter = self.tokenizer.segment_paragraph(text)
        self.words_no_stop_words = self.tokenizer.segment_paragraph(
            text, is_lower=True, is_use_stop_words=True,
            is_use_word_tags_filter=False)
        self.words_all_filters = self.tokenizer.segment_paragraph(
            text, is_lower=True, is_use_stop_words=True,
            is_use_word_tags_filter=False)

        if vertex_source == "all_filters":
            vertex_source_tmp = self.words_all_filters
        else:
            vertex_source_tmp = self.words_no_filter

        if edge_source == "no_stop_words":
            edge_source_tmp = self.words_no_stop_words
        else:
            edge_source_tmp = self.words_no_filter

        # 依据PageRank算法，计算每个词的重要程度
        self.keywords = compute_word_scores(vertex_source_tmp,
                                            edge_source_tmp,
                                            window_size=window_size,
                                            pagerank_config=pagerank_config)
        return self.keywords
