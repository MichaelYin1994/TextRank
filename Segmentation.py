#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:09:53 2020

@author:     zhuoyin94
@reference:  https://github.com/someus/
@email:      zhuoyin94@163.com
@github:     https://github.com/MichaelYin1994
"""

import re
import pkuseg

sentence_delimiters = ['?', '!', ';', '？',
                       '！', '。', '；', '……',
                       '…', '\n', '\t']
allow_word_tags = ['an', 'i', 'j', 'l', 'n',
                   'nr', 'nrfg', 'ns', 'nt',
                   'nz', 't', 'v', 'vd', 'vn', 'eng']
cop = re.compile(u"[^\u4e00-\u9faf^*^a-z^A-Z^0-9]")


class WordSegmentation():
    def __init__(self,
                 stop_words_vocab=None,
                 user_vocab=None,
                 is_lower=None,
                 is_use_stop_words=None,
                 is_use_word_tags_filter=None,
                 allow_word_tags=allow_word_tags,
                 delimiters=sentence_delimiters):
        # TODO: Add callback function

        self.stop_words = stop_words_vocab or []
        self.stop_words = [word.strip() for word in self.stop_words]
        self.stop_words = set(self.stop_words)

        self.is_lower = is_lower
        self.is_use_stop_words = is_use_stop_words
        self.is_use_word_tags_filter = is_use_word_tags_filter
        self.default_allow_word_tags = allow_word_tags
        self.default_user_vocab = user_vocab
        self.default_delimiters = set(delimiters)

        # TODO: Postag requires internet connection
        # FIX: Handling exception of internet failure
        self.seg = pkuseg.pkuseg(user_dict=user_vocab, postag=True)

    def segment_sentence(self, sentence,
                         is_lower=True,
                         is_use_stop_words=False,
                         is_use_word_tags_filter=False):

        if not isinstance(sentence, str):
            raise TypeError("Invalid input sentence type !")
        if self.is_lower:
            is_lower = self.is_lower
        if self.is_use_stop_words:
            is_use_stop_words = self.is_use_stop_words
        if self.is_use_word_tags_filter:
            is_use_word_tags_filter = self.is_use_word_tags_filter

        sentence_cutted = self.seg.cut(sentence)
        word_list = [item[0] for item in sentence_cutted]
        postag_list = [item[1] for item in sentence_cutted]

        # Filtering unsupport postag words
        if is_use_word_tags_filter:
            word_list = [word for i, word in enumerate(word_list) if postag_list[i] in self.default_allow_word_tags]

        # Filtering special tokens
        word_list = [word.strip() for word in word_list]
        word_list = [word for word in word_list if len(word) > 0]

        if is_lower:
            word_list = [word.lower() for word in word_list]

        # Filtering stop words
        if is_use_stop_words:
            word_list = [word for word in word_list if word not in self.stop_words]
        return word_list

    def segment_sentence_list(self, sentence_list,
                              is_lower=True,
                              is_use_stop_words=False,
                              is_use_word_tags_filter=False):

        if not isinstance(sentence_list, list):
            raise TypeError("Invalid input sentence list !")
        if self.is_lower:
            is_lower = self.is_lower
        if self.is_use_stop_words:
            is_use_stop_words = self.is_use_stop_words
        if self.is_use_word_tags_filter:
            is_use_word_tags_filter = self.is_use_word_tags_filter

        sentence_list_cutted = []
        for sentence in sentence_list:
            sentence_list_cutted.append(
                self.segment_sentence(sentence,
                                      is_lower,
                                      is_use_stop_words,
                                      is_use_word_tags_filter))
        return sentence_list_cutted

    def segment_paragraph(self, paragraph=None,
                          is_lower=True,
                          is_use_stop_words=False,
                          is_use_word_tags_filter=False):
        if not isinstance(paragraph, str):
            raise TypeError("Invalid input paragraph type !")
        if self.is_lower:
            is_lower = self.is_lower
        if self.is_use_stop_words:
            is_use_stop_words = self.is_use_stop_words
        if self.is_use_word_tags_filter:
            is_use_word_tags_filter = self.is_use_word_tags_filter

        # Step 1: Separate paragraph into sentences
        tmp = [paragraph]
        for sep in self.default_delimiters:
            sentence_list, tmp = tmp, []
            for sentence in sentence_list:
                tmp += sentence.split(sep)
        sentence_list = [s.strip() for s in sentence_list if len(s.strip()) > 0]

        # Step 2: Segment each sentence into word list
        sentence_list_cutted = self.segment_sentence_list(sentence_list,
                                                          is_lower,
                                                          is_use_stop_words,
                                                          is_use_word_tags_filter)
        return sentence_list_cutted


if __name__ == "__main__":
    pass
