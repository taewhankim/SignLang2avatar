import torch, numpy as np
import gluonnlp as nlp
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from gluonnlp.data import SentencepieceTokenizer
from kobert.data_utils.vocab_tokenizer import Tokenizer as BertTokenizer
from kobert.data_utils.pad_sequence import keras_pad_fn


class BertVocabClass:
    def __init__(self, vocab_file):
        self.vocab = nlp.vocab.BERTVocab.from_sentencepiece(
            vocab_file, padding_token="[PAD]", eos_token="[SEP]", bos_token="[CLS]")

        ptr_tokenizer = SentencepieceTokenizer(vocab_file)

        self.tokenizer = BertTokenizer(vocab=self.vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=72)
        self.usr_token_dict = {k:self.vocab.token_to_idx[k] for k in self.vocab.reserved_tokens}
        self.usr_token_dict[self.vocab.unknown_token] = self.vocab[self.vocab.unknown_token]

    def __len__(self):
        return len(self.vocab)

    def to_subword(self, data):
        return self.tokenizer.list_of_string_to_list_of_tokens(data)

    def vocab_to_num(self, data):
        try:
            temp = [self.vocab.token_to_idx[i] if i in self.vocab.idx_to_token else self.vocab.token_to_idx['[UNK]'] for i in data]
        except:
            temp = [self.vocab.token_to_idx[i] if i in self.vocab.idx_to_token else self.vocab.token_to_idx['<unk>'] for i in data]
        return temp

    def __call__(self, inputs):
        return self.tokenizer.list_of_string_to_list_token_ids(inputs)

    def num_to_vocab(self, data):
        return [self.vocab.idx_to_token[i] for i in data]  