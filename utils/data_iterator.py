import numpy as np
from utils import shuffle, data_utils
from utils.util import load_dict


'''
Much of this code is based on the data_iterator.py of
nematus project (https://github.com/rsennrich/nematus)
'''

'''
Python2 --> Python3
1) for key, idx in self.source_dict.items()   -->   for key, idx in list(self.source_dict.items()):

2) ss = [self.source_dict[w] if w in self.source_dict else data_utils.unk_token for w in ss]  -->
ss = [self.source_dict[bytes(w.encode())] if bytes(w.encode()) in self.source_dict
                      else data_utils.unk_token for w in ss]

3) Iterator class with __iter__,   next()  --> __next__()
'''

class BiTextIterator:
    """Simple Bitext iterator."""

    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=4,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 maxibatch_size=20):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = data_utils.fopen(source, 'r')
            self.target = data_utils.fopen(target, 'r')

        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for key, idx in list(self.source_dict.items()):
                if idx >= self.n_words_source:
                    del self.source_dict[key]

        if self.n_words_target > 0:
            for key, idx in list(self.target_dict.items()):
                if idx >= self.n_words_target:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])

    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[bytes(w.encode())] if bytes(w.encode()) in self.source_dict
                      else data_utils.unk_token for w in ss]

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[bytes(w.encode())] if bytes(w.encode()) in self.target_dict
                      else data_utils.unk_token for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target
                          else data_utils.unk_token for w in tt]

                if self.maxlen:
                    if len(ss) > self.maxlen and len(tt) > self.maxlen:
                        continue
                if self.skip_empty and (not ss or not tt):
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target


if __name__ == '__main__':
    print("loading dataset...")
    train_set = BiTextIterator(source='../data/sample.tok.en',
                               target='../data/sample.tok.fr',
                               source_dict='../data/sample.tok.en.json',
                               target_dict='../data/sample.tok.fr.json',
                               batch_size=4,
                               maxlen=34,
                               n_words_source=208,
                               n_words_target=216,
                               shuffle_each_epoch=False,
                               sort_by_length=True,
                               maxibatch_size=5)
    for source, target in train_set:
        print(source)
        print(target)