import torch
import numpy as np
import unicodedata
import sys

# A lot of the preprocessing inspired from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# I removed some bits that seemed unnecessary and added some extras

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

# Class that implements torch.utils.data.Dataset and so can be loaded using a dataloader
class LangDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

        # Initialise tokenziers for english and chinese respectively
        self.token_in = Tokenizer(by_word=True)
        self.token_out = Tokenizer(by_word=False)
        
        # open the dataset file from the specified path (read, magic string)
        lines = open("data/cmn.txt").readlines()

        # file is tab delimited so split by tab
        pairs = [[s for s in l.split('\t')] for l in lines]
        
        self.length = len(pairs)

        self.max_in = 0
        self.max_out = 0

        # for all pairs, add the sentence to the tokenizers
        # keep track of the maximum length seen so we can add the correct amount of padding later
        for p in pairs:
            self.token_in._add_sentence(p[0])
            if len(p[0]) > self.max_in:
                self.max_in = len(split_string(p[0], self.token_in.by_word))

            self.token_out._add_sentence(p[1])
            if len(p[1]) > self.max_out:
                self.max_out = len(split_string(p[1], self.token_out.by_word))

        # initialise data tensors of all pad
        self.data_src = torch.zeros(self.length, self.max_in)
        self.data_tgt = torch.zeros(self.length, self.max_out + 1)
        self.data_out = torch.zeros(self.length, self.max_out + 1)

        # again, enumerate through all pairs and add them to the tensors
        for i, p in enumerate(pairs):
            self._add_pair(i, p)

    # function to add a tokenized pair to the tensors at a given index
    def _add_pair(self, i, p):
        p[0] = remove_punctuation(p[0]).lower()
        p[1] = remove_punctuation(p[1]).lower()

        p0_len = len(split_string(p[0], self.token_in.by_word))
        p1_len = len(split_string(p[1], self.token_out.by_word)) + 1

        # tokenize the input pair and add to the tensor. Apply padding, sos and eos tokens as required
        self.data_src[i, :p0_len] = torch.tensor(self.token_in._sentence_to_index(p[0]))
        self.data_tgt[i, :p1_len] = torch.tensor([1] + self.token_out._sentence_to_index(p[1]))
        self.data_out[i, :p1_len] = torch.tensor(self.token_out._sentence_to_index(p[1]) + [2])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data_src[idx], self.data_tgt[idx], self.data_out[idx]
