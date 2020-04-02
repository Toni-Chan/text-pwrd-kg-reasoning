import numpy as np
import bcolz
words = []
idx = 0
word2idx = {}
glove_path = '/home/ubuntu/text-pwrd-kg-reasoning/'
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.200.dat', mode='w')

with open(f'{glove_path}/glove.6B.200d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
print(idx, ectors[1:], vectors.shape)
    
vectors = bcolz.carray(vectors[1:].reshape((400000, 200)), rootdir=f'{glove_path}/6B.200.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/6B.200_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/6B.200_idx.pkl', 'wb'))