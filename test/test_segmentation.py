#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Tue Dec 29 10:06:53 2020
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

import sys
import random
import pytest

import pkuseg
import numpy as np
if ".." not in sys.path:
    sys.path.append("..")

from segmentation import WordSegmentation

SENTENCE = ["根据列车运行速度计算安全行进距离",
            "数据平台和算法集市通过统一的数据总线和算法能力为上层智能智慧业务提供了全面的PaaS（平台即服务）服务",
            "交控科技还推出了列车远程瞭望系统的视距延伸装置——轨道星链",
            "公司总裁助理夏夕盛进行了“智慧单轨运行系统的发展及展望”主题演讲"]

SENTENCE_LIST = [["呼和浩特地铁2号线北起塔利东站",
                  "交控科技作为呼和地铁1号线的信号集成商和地铁1、2号线信号系统互联互通技术牵头方",
                  "为了让呼和浩特地铁更好的满足云化、智能化的运营要求"],
                 ["当“智能列车”到来前，可提前告知乘客“智能列车”的运行位置、不同车厢拥挤度、强冷/弱冷车厢",
                  "首次实现对于车内乘客晕倒、挥手求助等的检测和报警"]]

PARAGRAPH = "2020年10月21-23日，2020年“北京国际城市轨道交通展览会暨高峰论坛”在北京中国国际展览中心隆重举行。作为城市轨道交通信号系统的领军企业，交控科技股份有限公司（以下简称“交控科技”）携列车远程瞭望系统、天枢系统、智能列车乘客服务系统、无感改造、互联互通的CBTC系统、智慧管理、智慧培训等系统解决方案亮相，完整展示了智慧城轨的未来面貌，吸引大量业内专业人士及观众驻足观看交流。"

SENTENCE_ANONYMOUS = ["381 598 108 109 400 148 100 113",
                      "556 623 623 421 381 312",
                      "108 108 108 623 108 400",
                      "108 108 108 955, 623"]


def test_segment_sentence_list():
    # 一般性测试方法
    seg_tool = pkuseg.pkuseg(postag=True)
    seg = WordSegmentation(is_lower=False,
                           is_use_stop_words=False,
                           is_use_word_tags_filter=False)

    expected = []
    for sentence in SENTENCE:
        sentence_tmp = seg_tool.cut(sentence)
        sentence_tmp = [item[0] for item in sentence_tmp]
        expected.append(sentence_tmp)
    assert seg.segment_sentence_list(SENTENCE) == expected

    # 测试停用词滤除方法
    stop_words_vocab = ["根据", "了", "）", "（"]
    seg = WordSegmentation(is_lower=False,
                           is_use_stop_words=True,
                           is_use_word_tags_filter=False,
                           stop_words_vocab=stop_words_vocab)

    expected = []
    for sentence in SENTENCE:
        sentence_tmp = seg_tool.cut(sentence)
        sentence_tmp = [item[0] for item in sentence_tmp \
                        if item[0] not in stop_words_vocab]
        expected.append(sentence_tmp)
    assert seg.segment_sentence_list(SENTENCE) == expected

    # 测试基于词性的滤除方法
    allow_word_tags = ["n", "v"]
    seg = WordSegmentation(is_lower=False,
                           is_use_stop_words=False,
                           is_use_word_tags_filter=True,
                           allow_word_tags=allow_word_tags)

    expected = []
    for sentence in SENTENCE:
        sentence_tmp = seg_tool.cut(sentence)
        sentence_tmp = [item[0] for item in sentence_tmp \
                        if item[1] in allow_word_tags]
        expected.append(sentence_tmp)
    assert seg.segment_sentence_list(SENTENCE) == expected

    # 测试大小写是否正常转换
    seg = WordSegmentation(is_lower=True)

    expected = []
    for sentence in SENTENCE:
        sentence_tmp = seg_tool.cut(sentence)
        sentence_tmp = [item[0].lower() for item in sentence_tmp]
        expected.append(sentence_tmp)
    assert seg.segment_sentence_list(SENTENCE) == expected

if __name__ == "__main__":
    test_segment_sentence_list()