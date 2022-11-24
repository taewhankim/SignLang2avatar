from __future__ import absolute_import, division, print_function, unicode_literals

class Tokenizer:
    """ Tokenizer class"""

    def __init__(self, vocab, split_fn, pad_fn, maxlen):
        self._vocab = vocab 
        self._split = split_fn
        self._pad = pad_fn
        self._maxlen = maxlen

    # def split(self, string: str) -> list[str]:
    def split(self, string):
        tokens = self._split(string)
        return tokens

    # def transform(self, list_of_tokens: list[str]) -> list[int]:
    def transform(self, tokens):
        indices = self._vocab.to_indices(tokens)
        pad_indices = self._pad(indices, pad_id=0, maxlen=self._maxlen) if self._pad else indices
        return pad_indices

    # def split_and_transform(self, string: str) -> list[int]:
    def split_and_transform(self, string):
        return self.transform(self.split(string))

    @property
    def vocab(self):
        return self._vocab


    def transform_token2idx(self, token, show_oov=False):
        try:
            return self._vocab.token_to_idx[token]
        except:
            if show_oov is True:
                print("key error: " + str(token))
            token = self._vocab.unknown_token
            return self._vocab.token_to_idx[token]

    def transform_idx2token(self, idx):
        try:
            return self._vocab.idx_to_token[idx]
        except:
            print("key error: " + str(idx))
            idx = self._vocab.token_to_idx[self._vocab.unknown_token]
            return self._vocab.idx_to_token[idx]


    def list_of_tokens_to_list_of_token_ids(self, X_token_batch):
        X_ids_batch = []
        for X_tokens in X_token_batch:
            X_ids_batch.append([self.transform_token2idx(X_token) for X_token in X_tokens])
        return X_ids_batch

    def list_of_string_to_list_of_tokens(self, X_str_batch):
        X_token_batch = [self._split(X_str) for X_str in X_str_batch]
        return X_token_batch

    def list_of_tokens_to_list_of_token_ids(self, X_token_batch):
        X_ids_batch = []
        for X_tokens in X_token_batch:
            X_ids_batch.append([self.transform_token2idx(X_token) for X_token in X_tokens])
        return X_ids_batch

    def list_of_string_to_list_token_ids(self, X_str_batch):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        X_ids_batch = self.list_of_tokens_to_list_of_token_ids(X_token_batch)

        return X_ids_batch

    def list_of_string_to_arr_of_pad_token_ids(self, X_str_batch, add_start_end_token=False):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        # print("X_token_batch: ", X_token_batch)
        if add_start_end_token is True:
            return self.add_start_end_token_with_pad(X_token_batch)
        else:
            X_ids_batch = self.list_of_tokens_to_list_of_token_ids(X_token_batch)
            pad_X_ids_batch = self._pad(X_ids_batch, pad_id=self._vocab['[PAD]'], maxlen=self._maxlen)

        return pad_X_ids_batch

    def list_of_tokens_to_list_of_cls_sep_token_ids(self, X_token_batch):
        X_ids_batch = []
        for X_tokens in X_token_batch:
            X_tokens = [self._vocab['[CLS]']] + X_tokens + [self._vocab['[SEP]']]
            X_ids_batch.append([self.transform_token2idx(X_token) for X_token in X_tokens])
        return X_ids_batch

    def list_of_string_to_arr_of_cls_sep_pad_token_ids(self, X_str_batch):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        X_ids_batch = self.list_of_tokens_to_list_of_cls_sep_token_ids(X_token_batch)
        pad_X_ids_batch = self._pad(X_ids_batch, pad_id=self._vocab['[PAD]'], maxlen=self._maxlen)

        return pad_X_ids_batch

    def list_of_string_to_list_of_cls_sep_token_ids(self, X_str_batch):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        X_ids_batch = self.list_of_tokens_to_list_of_cls_sep_token_ids(X_token_batch)

        return X_ids_batch

    def add_start_end_token_with_pad(self, X_token_batch):
        dec_input_token_batch = [[self._vocab['[SOS]']] + X_token for X_token in X_token_batch]
        dec_output_token_batch = [X_token + [self._vocab['[EOS]']] for X_token in X_token_batch]

        dec_input_token_batch = self.list_of_tokens_to_list_of_token_ids(dec_input_token_batch)
        pad_dec_input_ids_batch = self._pad(dec_input_token_batch, pad_id=self._vocab['[PAD]'], maxlen=self._maxlen)

        dec_output_ids_batch = self.list_of_tokens_to_list_of_token_ids(dec_output_token_batch)
        pad_dec_output_ids_batch = self._pad(dec_output_ids_batch, pad_id=self._vocab['[PAD]'], maxlen=self._maxlen)
        return pad_dec_input_ids_batch, pad_dec_output_ids_batch

    def decode_token_ids(self, token_ids_batch):
        list_of_token_batch = []
        for token_ids in token_ids_batch:
            token_token = [self.transform_idx2token(token_id) for token_id in token_ids]
            # token_token = [self._vocab[token_id] for token_id in token_ids]
            list_of_token_batch.append(token_token)
        return list_of_token_batch