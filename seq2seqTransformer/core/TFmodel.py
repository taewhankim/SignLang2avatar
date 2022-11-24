import torch
import torch.nn as nn
import math
from torch.nn.functional import softmax
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),"../"))
from .data_classes import suwhaVocabClass
from torch import Tensor
from torch.nn import (Transformer ,TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class TF_conf:
    def __init__(self, src_vocab, tgt_vocab,
                    emb_size=256, nhead=8, 
                    ffn_hid_dim=512, num_encoder_layers=8, 
                    num_decoder_layers=8, dropout=0.1):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)
        self.emb_size = emb_size
        self.nhead = nhead
        self.ffn_hid_dim = ffn_hid_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

    def load_Seq2SeqTF_from_conf(self):
        transformer = Seq2SeqTransformer(self.num_encoder_layers, self.num_decoder_layers,
                                 self.emb_size, self.src_vocab_size, self.tgt_vocab_size,
                                 self.ffn_hid_dim, dropout=self.dropout, 
                                 src_vocab = self.src_vocab, tgt_vocab=self.tgt_vocab)
        return transformer

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1, nhead=8, 
                 src_vocab:suwhaVocabClass=None, tgt_vocab:suwhaVocabClass=None):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=nhead,\
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.to(DEVICE)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor): 
                
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        #
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        #
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)


    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

    def generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

        src_padding_mask = (src == 0).transpose(0, 1)
        tgt_padding_mask = (tgt == 0).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def greedy_decode(self, src, src_mask, max_len, start_symbol, k=5):
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        memory = self.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        for i in range(max_len-1):
            memory = memory.to(DEVICE)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                                        .type(torch.bool)).to(DEVICE)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.item()
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == DEVICE:
              break
        return ys

    def translate_greedy(self, src):
        tokens = [2] + self.tgt_vocab(src) + [3]
        num_tokens = len(tokens)
        src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens + 5, start_symbol=2).flatten().tolist()
        return " ".join(self.tgt_vocab.num_to_vocab(tgt_tokens)).replace("[BOS]", "").replace("[EOS]", "").replace("[UNK]", "") , tgt_tokens

    def translate_beam(self, src):
        tokens = [2] + self.tgt_vocab(src) + [3]
        num_tokens = len(tokens)
        src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
        Beam = BEAM_SEARCH(self, start_sign=self.tgt_vocab.vocab_num_dict['[UNK]'], end_sign=self.tgt_vocab.vocab_num_dict['[BOS]'], k=5)
        result = Beam.beam_search(src)
        result=result[0]
        return "".join(self.tgt_vocab.num_to_vocab(result)).replace("[BOS]", "").replace("[EOS]", "").replace("[UNK]", " ").split(" ")#, result

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class BEAM_SEARCH:
    def __init__(self, model:Seq2SeqTransformer, start_sign=2, end_sign=3, k=5):
        self.topks = None
        self.next = None
        self.model = model
        self.k = k
        self.END=end_sign
        self.START=start_sign


    def beam_search(self, src):
        src_mask = (torch.zeros(src.shape[0], src.shape[0])).type(torch.bool)
        max_len= src.shape[0] + 5
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)
        memory = self.model.encode(src, src_mask).to(DEVICE)
        queue = []
        flag_queue = [[self.START]]
        score_queue = []
        fscore_queue = [1.]
        while queue != flag_queue:
            queue = flag_queue
            score_queue = fscore_queue
            local_topks = []
            local_scores = []
            for q, s in zip(queue, score_queue):
                if q[-1] == self.END or len(q) >= max_len:
                    continue
                ys = torch.tensor(q).unsqueeze(1).type(torch.long).to(DEVICE)
                local_s, local_k = self.deep_inside(ys, memory)
                for s, k in zip(local_s, local_k):
                    local_topks.append(q+[k])
                    local_scores.append(s* -math.log(s))
                    
            if len(local_topks) > self.k:
                _, local_top_idx = torch.topk(torch.tensor(local_scores), k=self.k, dim=0)
                local_topks = [local_topks[i] for i in local_top_idx]
                local_scores = [local_scores[i] for i in local_top_idx]
            if len(local_topks) > 0:
                flag_queue = local_topks
        return flag_queue

    def deep_inside(self, ys, memory):
        tgt_mask = (self.model.generate_square_subsequent_mask(ys.size(0))
                                        .type(torch.bool)).to(DEVICE)
        out = self.model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = self.model.generator(out[:, -1])
        prob_mat = softmax(prob, dim=1)
        pval, topk = torch.topk(prob_mat, k=self.k, dim=1)
        return pval[0].detach().tolist(), topk[0].detach().tolist()