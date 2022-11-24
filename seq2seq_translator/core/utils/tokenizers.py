from typing import Any, Dict, List, Optional
from transformers.tokenization_utils import AddedToken
from transformers import XLNetTokenizer, PreTrainedTokenizerFast
from transformers import SPIECE_UNDERLINE
import torch, re, pandas as pd
from typing import List, Union
import sentencepiece as spm
import re, os
from konlpy.tag import Twitter
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

twt = Twitter()


class InputToken:
    def __init__(
        self,
        input_ids: list,
        attention_mask: list,
        token_type_ids: list,
        input_len: list,
        device,
        pad_token_id=1,
    ):
        self.input_ids = torch.tensor(input_ids, device=device)
        self.attention_mask = torch.tensor(
            attention_mask, dtype=torch.float32, device=device
        )
        self.token_type_ids = torch.tensor(token_type_ids, device=device)
        self.input_len = input_len
        self.device = device
        self.pad_token_id = pad_token_id

    def update(self, seq):
        # seq : (batch, seq_len)
        padded_list = add_padding(seq, pad_id=self.pad_token_id)
        maxlen = max([len(x) for x in seq])

        self = InputToken(
            input_ids=padded_list,
            attention_mask=[([1] * len(r)) + ([0] * (maxlen - len(r))) for r in seq],
            token_type_ids=[[0] * maxlen for i in range(len(seq))],
            input_len=[len(r) for r in seq],
            device=self.device,
        )
        return self

    def __str__(self):
        return f"'input_ids':{self.input_ids}, 'attention_mask':{self.attention_mask}, \
                    'token_type_ids':{self.token_type_ids}, 'input_len':{self.input_len}"

    @property
    def tgt_token(self):
        input_ids = self.input_ids.tolist()
        input_len = self.input_len
        seq = [ids[: l - 1] for l, ids in zip(input_len, input_ids)]
        padded_list = add_padding(seq, self.pad_token_id)
        maxlen = max([len(x) for x in seq])

        return InputToken(
            input_ids=padded_list,
            attention_mask=[([1] * len(r)) + ([0] * (maxlen - len(r))) for r in seq],
            token_type_ids=[[0] * maxlen for i in range(len(seq))],
            input_len=[len(r) for r in seq],
            device=self.device,
        )


class Tokenizer:
    def __init__(self, conf):
        def get_t(c):
            if c.name == "KoBertVocab":
                return KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
            elif c.name == "custom":
                idx_to_word, word_to_idx = get_vocab(c.vocab_dir)
                return lambda x: custom_tokenizer(x, word_to_idx)
            elif c.name == "sentencepiece":
                tokenizer = spm.SentencePieceProcessor(model_file=c.sentencepiece_dir)
                return tokenizer
            elif c.name == "sentencepiece_from_txt":
                c.name = "sentencepiece"
                tokenizer = SentencePiece(c.max_vocab_size).make_sentencepiece(
                    c.input_file, "./"
                )
            elif c.name == "KoGPTVocab":
                return PreTrainedTokenizerFast.from_pretrained(
                    "skt/kogpt2-base-v2",
                    bos_token="</s>",
                    eos_token="</s>",
                    unk_token="<unk>",
                    pad_token="<pad>",
                    mask_token="<mask>",
                    cache_dir="/mnt/workdir/sign/seq2seq_translator/core/gpt/pretrained",
                )
            else:
                print(f"error: {c.name} cannot load tokenizer!!")
                exit()

        self.tokenizer = get_t(conf)
        self.conf = conf

    def decode(self, token_lists):
        if self.conf.name in ["KoBertVocab", "KoGPTVocab"]:
            result = []
            for tokens in token_lists:
                result.append(self.tokenizer._decode(tokens))
            return result
        elif self.conf.name in ["sentencepiece", "sentencepiece_from_txt"]:
            result = self.tokenizer.Decode(token_lists)

        return result

    def encode(
        self,
        input_str,
        device="cuda" if torch.cuda.is_available() else "cpu",
        is_nl=False,
        remove_stokens=True,
        num_eng_blank=True,
    ):
        # input_str = [
        #     preproc(
        #         s,
        #         is_nl=is_nl,
        #         remove_stokens=remove_stokens,
        #         num_eng_blank=num_eng_blank,
        #     )
        #     for s in input_str
        # ]

        if self.conf.name in ["KoBertVocab", "KoGPTVocab"]:
            if self.conf.name == "KoGPTVocab":
                temp = []
                for line in input_str:
                    temp.append(
                        self.tokenizer.bos_token + line + self.tokenizer.eos_token
                    )
                input_str = temp
            
            result = self.tokenizer(input_str, padding=True, return_length=True)
            temp = torch.tensor(result["input_ids"])
            length = [
                len(l) - l.count(self.tokenizer.pad_token_id)
                for l in result["input_ids"]
            ]
            tokenized_sentence = InputToken(
                input_ids=result["input_ids"],
                attention_mask=result["attention_mask"],
                token_type_ids=result["token_type_ids"],
                input_len=length,
                device=device,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            return tokenized_sentence
        elif self.conf.name == "custom":
            return self.tokenizer(list(input_str))
        elif self.conf.name in ["sentencepiece", "sentencepiece_from_txt"]:

            result_list = []
            for string in input_str:
                result_list.append(
                    self.tokenizer.encode(string, add_bos=True, add_eos=True)
                )
            padded_list = add_padding(result_list)
            maxlen = max([len(x) for x in result_list])

            tokenized_sentence = InputToken(
                input_ids=padded_list,
                attention_mask=[
                    ([1] * len(r)) + ([0] * (maxlen - len(r))) for r in result_list
                ],
                token_type_ids=[[0] * maxlen for i in range(len(result_list))],
                input_len=[len(r) for r in result_list],
                device=device,
            )

            return tokenized_sentence

    def batch_tokenize(self, batch_sentence):
        """
        input:
            batch_sentence : list of sentences

        output:
            list of tokenized sentences
        """
        batch_tokenized_sentence = list()
        for sentence in batch_sentence:
            tokenized_sentence = self.tokenizer.tokenize(sentence)
            tokenized_sentence = [i.replace("_", "") for i in tokenized_sentence]
            batch_tokenized_sentence.append(tokenized_sentence)
        return batch_tokenized_sentence

    def len_tokenized_sentence(self, tokenized_sentence):
        if self.conf.name == "KoBertVocab":
            return len(tokenized_sentence["attention_mask"])
        elif self.conf.name == "custom":
            return len(tokenized_sentence[0])
        elif self.conf.name == "sentencepiece":
            return len(tokenized_sentence)

    @property
    def get_all_usr_tokens(self):
        if self.conf.name in ["KoBertVocab", "KoGPTVocab"]:
            return self.tokenizer.all_special_tokens
        elif self.conf.name in ["sentencepiece", "sentencepiece_from_txt"]:
            return self.tokenizer.USER_TOKENS


def get_tokenizer_from_conf(conf):
    config = conf.tokenizer
    if config.encoder.name == config.decoder.name:
        en_token = Tokenizer(config.encoder)
        return en_token, en_token
    return Tokenizer(config.encoder), Tokenizer(config.decoder)


def custom_tokenizer(input_string_list, word_to_idx):  # batch x length
    assert isinstance(
        input_string_list, List
    ), "input string list should be list, size : batch x length"
    result_list = []

    for string in input_string_list:
        rmv_string = clean_text(string).split()
        rmv_string = ["[START]"] + rmv_string + ["[END]"]
        result_list.append(rmv_string)

    max_length = max([len(string) for string in result_list])

    # padding
    for index in range(len(result_list)):
        result_list[index] = result_list[index] + ["[PAD]"] * (
            max_length - len(result_list[index])
        )

        # tokenize
        result_list[index] = [word_to_idx[i] for i in result_list[index]]

    return result_list


def create_sentencepiece(
    sentencepiece_dir,
    input_dir,
    model_prefix,
    vocab_size,
    character_coverage,
    model_type,
):
    spm.SentencePieceTrainer.train(
        input=input_dir,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=1,
        unk_id=0,
        bos_id=2,
        eos_id=3,
    )


def add_padding(input: List, pad_id=1):
    input_string = input.copy()
    max_len = max([len(i) for i in input_string])

    for idx in range(len(input_string)):
        input_string[idx] = input_string[idx] + [pad_id] * (
            max_len - len(input_string[idx])
        )
    return input_string


special_token = ["[START]", "[END]", "[PAD]"]


def clean_text(input_string):
    text_rmv = remove_special_token(input_string)
    text_rmv = spacing_numbers(text_rmv)
    text_rmv = " ".join(text_rmv.split())
    return text_rmv


def remove_special_token(input_string):
    text_rmv = re.sub("[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`'…》\”\“\’·]", " ", input_string)
    return text_rmv


def spacing_numbers(input_string):
    input_string = re.sub("0", "0 ", input_string)
    input_string = re.sub("1", "1 ", input_string)
    input_string = re.sub("2", "2 ", input_string)
    input_string = re.sub("3", "3 ", input_string)
    input_string = re.sub("4", "4 ", input_string)
    input_string = re.sub("5", "5 ", input_string)
    input_string = re.sub("6", "6 ", input_string)
    input_string = re.sub("7", "7 ", input_string)
    input_string = re.sub("8", "8 ", input_string)
    input_string = re.sub("9", "9 ", input_string)
    return input_string


def create_vocab(**kwargs):
    excel_file = pd.read_excel(kwargs["root_dir"], engine="openpyxl")
    sign_language = excel_file["수어"].tolist()
    vocab_list = []
    for sign in sign_language:
        sign = clean_text(sign)
        split_sign = sign.split()
        for word in split_sign:
            if word in vocab_list:
                pass
            else:
                vocab_list.append(word)
    vocab_list = special_token + vocab_list
    df = pd.DataFrame(vocab_list)
    df.to_csv(kwargs["vocab_dir"], index=False, header=False)


def get_vocab(vocab_dir):
    vocab_file = pd.read_csv(vocab_dir, names=["word"])
    vocab = vocab_file["word"]
    idx_to_word, word_to_idx = vocab, key_value_change(vocab)
    return idx_to_word, word_to_idx


def key_value_change(input_dic):
    return dict((value, key) for (key, value) in input_dic.items())


class KoBERTTokenizer(XLNetTokenizer):
    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )
        self._pad_token_type_id = 0

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLNet sequence has the following format:
        - single sequence: ``<cls> X <sep>``
        - pair of sequences: ``<cls> A <sep> B <sep>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        text = self.preprocess_text(text)
        pieces = self.sp_model.encode(text, out_type=str, **self.sp_model_kwargs)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, "")
                )
                if (
                    piece[0] != SPIECE_UNDERLINE
                    and cur_pieces[0][0] == SPIECE_UNDERLINE
                ):
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLNet sequence has the following format:
        - single sequence: ``<cls> X <sep> ``
        - pair of sequences: ``<cls> A <sep> B <sep>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
        sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]


class SentencePiece:
    def __init__(self, max_vocab_size=2000):
        self.vocab_size = None
        self.vocab_path = None
        self.num_vocab_dict = None
        self.vocab_num_dict = None
        self.max_vocab_size = max_vocab_size
        self.spm = spm.SentencePieceProcessor()
        self.USER_TOKENS = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[SEP]": 4,
            "[CLS]": 5,
            "[MASK]": 6,
        }

    def load_spm(self, vocab_path=None):
        if vocab_path == None:
            vocab_path = self.vocab_path
        self.spm.load(vocab_path + ".model")
        with open(vocab_path + ".vocab", "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.vocab_size = len(self.spm)
        self.vocab_path = vocab_path
        self.num_vocab_dict = {
            i + 1: text.split()[0].strip() for i, text in enumerate(lines)
        }
        self.vocab_num_dict = dict(
            zip(self.num_vocab_dict.values(), self.num_vocab_dict.keys())
        )
        return self.spm

    def make_sentencepiece(
        self,
        input_file,
        output_folder=".",
        output_name="vocab",
        only_file=False,
        is_nl=False,
        remove_stokens=True,
        num_eng_blank=True,
    ):
        sp_input_file = output_folder + "/sptemp.txt"
        output_folder = output_folder + "/vocab"
        with open(input_file, "r") as inf:
            with open(sp_input_file, "w") as spf:
                lines = inf.readlines()
                for line in lines:
                    if re.findall("[\(\[\{<]", line):
                        continue
                    try:
                        ta, tr = line.strip().strip("\n").split("|")
                    except:
                        print(line.strip().strip("\n").split("|"))
                        exit()
                    # ta = preproc(
                    #     ta,
                    #     is_nl=is_nl,
                    #     remove_stokens=remove_stokens,
                    #     num_eng_blank=num_eng_blank,
                    # )
                    # tr = preproc(
                    #     tr,
                    #     is_nl=is_nl,
                    #     remove_stokens=remove_stokens,
                    #     num_eng_blank=num_eng_blank,
                    # )
                    write_line = ta + "|" + tr + "\n"
                    if write_line != "|\n":
                        spf.write(write_line)
        if only_file:
            return sp_input_file
        vocab_size = self.max_vocab_size
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        vocap_path = os.path.abspath(os.path.join(output_folder, output_name))
        self.vocab_size = vocab_size
        spm.SentencePieceTrainer.train(
            f"--input={sp_input_file} --model_prefix={vocap_path} --vocab_size={vocab_size + 7}"
            + " --model_type=bpe"
            + " --max_sentence_length=999999"
            + " --pad_id=0 --pad_piece=[PAD]"  # 문장 최대 길이
            + " --unk_id=1 --unk_piece=[UNK]"  # pad (0)
            + " --bos_id=2 --bos_piece=[BOS]"  # unknown (1)
            + " --eos_id=3 --eos_piece=[EOS]"  # begin of sequence (2)
            + " --user_defined_symbols=[SEP],[CLS],[MASK]"  # end of sequence (3)
        )  # 사용자 정의 토큰
        self.vocab_path = vocap_path
        return self.load_spm(self.vocab_path)

    def __len__(self):
        return self.vocab_size + 7

    def to_subword(self, data):
        return self.spm.encode_as_pieces(data)

    def __call__(self, data):
        self.spm.encode_as_pieces(data)
        return self.vocab_to_num(data)

    def vocab_to_num(self, data):
        try:
            temp = [
                self.vocab_num_dict[i]
                if i in self.vocab_num_dict.keys()
                else self.vocab_num_dict["[UNK]"]
                for i in data
            ]
        except:
            temp = [
                self.vocab_num_dict[i]
                if i in self.vocab_num_dict.keys()
                else self.vocab_num_dict["<unk>"]
                for i in data
            ]

        return temp

    def num_to_vocab(self, data):
        return [self.num_vocab_dict[i] for i in data]


def preproc(line, is_nl=False, remove_stokens=True, num_eng_blank=True):
    # pre1 괄호제거
    pre1 = "[\(\[\<\]\)][ㄱ-ㅎ가-힣0-9a-zA-Z,. \|\n?!:]+[\(\[\<\>\]\)]"
    # pre2 특문제거
    pre2 = "[^ㄱ-ㅎ가-힣0-9a-zA-Z,. \|\n?!:_]"
    # pre34 대문자, 숫자 떨어뜨리기 + 앞 뒤 스페이스
    pre3 = "([A-Z][^a-zA-Z ]|[^a-zA-Z ][A-Z])"
    pre4 = "[A-Z]{2,100}|[0-9]{2,}"
    # pre5 = 영단어 앞, 뒤 스페이스
    pre5 = "[A-Z][a-z]+|[a-z]+|[.,!?~:]+|[0-9]+"
    # pre6 = 스페이스 2회 이상 -> 1회
    pre6 = "[ ]{2,}"

    l = line.strip("\n").strip()
    l = re.sub(pre1, "", l)
    if remove_stokens:
        l = re.sub(pre2, " ", l)

    if num_eng_blank:
        uppers = re.findall(pre3, l)
        for upp in uppers:
            re_case = " " + " ".join([u for u in upp]) + " "
            l = l.replace(upp, re_case)

        uppers = [1, 1]
        while len(uppers) != 0:
            uppers = re.findall(pre4, l)
            for upp in uppers:
                re_case = " " + " ".join([u for u in upp]) + " "
                l = l.replace(upp, re_case)

    # 형태소 분석에 의한 split
    if is_nl:
        tags = twt.pos(l)
        l = " ".join([i for i, j in tags])

    if num_eng_blank:
        words = re.findall(pre5, l)
        for w in words:
            l = l.replace(w, " " + w + " ")

    spaces = [1, 1]
    while len(spaces) != 0:
        spaces = re.findall(pre6, l)
        for s in spaces:
            l = l.replace(s, " ")

    return l.strip()
