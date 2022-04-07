import os
import tensorflow as tf
import numpy as np

print('Hello')

text = open('./tiny-shakespeare.txt', 'rb').read()

# print(text[0:100])

vocab = sorted(set(text))
print('{} unique chars..'.format(len(vocab)))

char2idx = {unique:idx for idx, unique in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[char] for char in text])

print('{')
for char, _ in zip(char2idx, range(65)):
    print(' {:4s}: {:3d},'.format(chr(char), char2idx[char]))
print('....\n')

print('{} ----> chars mapped to int ---> {}'.format(text[:13],
                                                    text_as_int[:13]))
