# _*_ coding:utf-8 _*_
# author: jihe
# time: 2020/5/2 17:09
# name: RNN
# From
# 是一种适宜于处理序列数据的神经网络，被广泛用于语言模型、文本生成、机器翻译等
import tensorflow as tf
import numpy as np


class DataLoader():
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt',
                                       origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')  # 网址挂了......
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))  # 出现的字符
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))  # enumerate i是索引值，c是打印元素（self.chars）是遍历对象
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]  # 小说字符转换成数字化

    # seq_length窗口长度，batch_size获取样本数 next_char = []样本的标签
    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index + seq_length])
            next_char.append(self.text[index + seq_length])
        return np.array(seq), np.array(next_char)  # [batch_size, seq_length], [batch，size]


data_loader = DataLoader()
batch = data_loader.get_batch(3, 20)
print(batch)
