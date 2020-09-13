import torch
import numpy as np
import unicodedata
import sys

# object used to strip all punctuation in unicode range
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))

# helper function to remove punctuation
def remove_punctuation(text):
    return text.translate(tbl)

# helper function to split strings either by words or by character
def split_string(s, by_word):
    return s.split(' ') if by_word else list(s)

# Class that accepts sentences and tokenizes words within them
# TODO: Unknown token for unknown or low frequency words?
class Tokenizer:
    def __init__(self, by_word=True):
        self.by_word = by_word
        self.word_index = {}
        self.word_count = {}
        self.index_word = {0: "PAD", 1: "SOS", 2: "EOS"} # prepopulate with special tokens
        self.nb_words = 3

    def _add_sentence(self, s):
        s = remove_punctuation(s)
        s = s.lower()
        for w in split_string(s, self.by_word):
            self._add_word(w)

    def _add_word(self, w):
        if not w in self.word_index: # if this is a new token
            self.word_index[w] = self.nb_words
            self.word_count[w] = 1
            self.index_word[self.nb_words] = w
            self.nb_words += 1
        else: # if we have seen this token before
            self.word_count[w] += 1

    # Takes a sentence and converts it to a list of tokens
    def _sentence_to_index(self, s):
        s = remove_punctuation(s).lower()
        return [self.word_index[w] for w in (split_string(s, self.by_word))]

class LangDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

        self.token_in = Tokenizer(by_word=True)
        self.token_out = Tokenizer(by_word=False)

        lines = open("data/cmn.txt").readlines()
        pairs = [[s for s in l.split('\t')] for l in lines]
        
        self.length = len(pairs)

        self.max_in = 0
        self.max_out = 0

        for p in pairs:
            self.token_in._add_sentence(p[0])
            if len(p[0]) > self.max_in:
                self.max_in = len(split_string(p[0], self.token_in.by_word))

            self.token_out._add_sentence(p[1])
            if len(p[1]) > self.max_out:
                self.max_out = len(split_string(p[1], self.token_out.by_word))

        self.data_in = torch.zeros(self.length, self.max_in + 2)
        self.data_out = torch.zeros(self.length, self.max_out + 3)

        for i, p in enumerate(pairs):
            self._add_pair(i, p)

    def _add_pair(self, i, p):
        p[0] = remove_punctuation(p[0]).lower()
        p[1] = remove_punctuation(p[1]).lower()

        p0_len = len(split_string(p[0], self.token_in.by_word)) + 2
        p1_len = len(split_string(p[1], self.token_out.by_word)) + 3

        self.data_in[i, :p0_len] = torch.tensor([1] + [self.token_in.word_index[w] for w in (split_string(p[0], self.token_in.by_word))] + [2])
        self.data_out[i, :p1_len] = torch.tensor([1] + [self.token_out.word_index[w] for w in (split_string(p[1], self.token_out.by_word))] + [2, 0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data_in[idx], self.data_out[idx]
