import numpy as np
import bcolz
glove_path = '/home/ubuntu/text-pwrd-kg-reasoning/'
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.200.dat', mode='w')

words = []
idx = 0
word2idx = {}

with open(f'{glove_path}/glove.6B.200d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400001, 200)), rootdir=f'{glove_path}/6B.200.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/6B.200_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/6B.200_idx.pkl', 'wb'))