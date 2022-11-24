from turtle import forward
from typing import Optional
from transformers import BertModel
from torch.nn import *
from seq2seq_translator.core.kobert import get_pretrained_kobert
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from torch import Tensor
import copy
import math, numpy as np, torch, os
import torch.nn.functional as F

cur_path = os.path.abspath(os.path.dirname(__file__))


def PreKoBert(freeze=False, device="cuda") -> BertModel:

    model = get_pretrained_kobert(device=device)
    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    return model


def TFDecoder(
    emb_size=768,
    nhead=8,
    dim_feedforward=512,
    num_layers=8,
    dropout=0.2,
    activation="relu",
):
    decoder_layer = TransformerDecoderLayer(
        d_model=emb_size,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )  # batch_first=True
    return TransformerDecoder(decoder_layer, num_layers=num_layers)


def GFDecoder(
    emb_size=768,
    nhead=8,
    dim_feedforward=512,
    num_layers=8,
    dropout=0.2,
    activation="relu",
):
    decoder_layer = TransformerDecoderLayer(
        d_model=emb_size,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )  # batch_first=True
    return GraformerDecoder(decoder_layer, num_layers=num_layers)


def TFEncoder(
    emb_size=768,
    nhead=16,
    dim_feedforward=512,
    num_layers=8,
    dropout=0.2,
    activation="relu",
):
    decoder_layer = TransformerEncoderLayer(
        d_model=emb_size,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    return TransformerEncoder(decoder_layer, num_layers=num_layers)


class TokenEmbedding(Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class GraformerDecoder(Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(GraformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:

        output = tgt

        for mod in self.layers:
            output = (
                mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                + tgt
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class BertPositionalEmb(Module):
    def __init__(
        self,
        emb_size: int,
        dropout=0.2,
        max_len: int = 2000,
        pretrained=False,
    ):
        super(BertPositionalEmb, self).__init__()
        self.emb = PreKoBert(pretrained=pretrained, freeze=True)

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, x):
        x, mask = x
        with torch.no_grad():
            x = self.emb(x, mask)[0].transpose(0, 1)
        return self.dropout(x + self.pos_embedding[: x.size(0), :])


class PositionalEncoding(Module):
    def __init__(
        self,
        emb_size: int,
        dropout,
        vocab_size=8002,
        max_len: int = 2000,
    ):
        super(PositionalEncoding, self).__init__()
        self.emb = TokenEmbedding(vocab_size, emb_size)

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, x):
        x = self.emb(x)
        return self.dropout(x + self.pos_embedding[: x.size(0), :])


class PoSEmbedding(Module):
    def __init__(
        self,
        embedding_name,
        embedding_dim,
        input_dim,
        padding_idx,
    ):
        super(PoSEmbedding, self).__init__()

        if embedding_name == "Embedding":
            self.pos_embedding = Embedding(
                num_embeddings=input_dim,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
            )
        else:
            raise Exception("we only have embedding name = Embedding")

    def forward(self, batch_pos_list):
        batch_tensor_result = None

        for batch_idx in range(len(batch_pos_list)):
            tensor_result = None

            for token_idx in range(len(batch_pos_list[batch_idx])):
                if token_idx == 0:
                    pos_embedding = self.pos_embedding(
                        torch.LongTensor(batch_pos_list[batch_idx][token_idx]).to(
                            "cuda"
                        )
                    )
                    if pos_embedding.size()[0] != 1:
                        pos_embedding = torch.sum(pos_embedding, dim=0).unsqueeze(dim=0)
                    tensor_result = pos_embedding
                else:
                    pos_embedding = self.pos_embedding(
                        torch.LongTensor(batch_pos_list[batch_idx][token_idx]).to(
                            "cuda"
                        )
                    )
                    if pos_embedding.size()[0] != 1:
                        pos_embedding = torch.sum(pos_embedding, dim=0).unsqueeze(dim=0)
                    tensor_result = torch.cat([tensor_result, pos_embedding], dim=0)

            tensor_result = tensor_result.unsqueeze(dim=0)
            if batch_idx == 0:
                batch_tensor_result = tensor_result
            else:
                batch_tensor_result = torch.cat(
                    [batch_tensor_result, tensor_result], dim=0
                )
        return batch_tensor_result


def LinearGenerator(
    in_features=768,
    out_features=8002,
):
    return Sequential(
        Linear(
            in_features=in_features,
            out_features=out_features,
        ),
        ReLU(),
    )


def Lin(
    in_features=768,
    out_features=8002,
    dim=2,
):
    return Sequential(
        # Softmax(dim),
        Linear(
            in_features=in_features,
            out_features=out_features,
        ),
    )


def GraformerGenerator(
    in_features=768,
    out_features=8002,
    dim=2,
):
    return Sequential(
        Linear(
            in_features=in_features,
            out_features=out_features,
        ),
        LogSoftmax(dim),
    )


class ConcatLinearGenerator(Module):
    def __init__(
        self,
        in_features=768,
        out_features=51200,
    ):
        super(ConcatLinearGenerator, self).__init__()
        self.l1 = Linear(in_features, in_features)
        self.l2 = Linear(in_features, in_features)
        self.c = Linear(in_features, out_features)
        self.relu = ReLU()

    def forward(self, xs: torch.Tensor):
        x1, x2 = xs.chunk(2, dim=0)
        x1 = self.l1(x1)
        x2 = self.l2(x2)
        x = x1 * x2
        x = self.c(x)
        return self.relu(x)


def KoGPT2(
    cache_dir=os.path.join(cur_path, "..", "gpt", "pretrained_gpt"),
    weight="skt/kogpt2-base-v2",
    freeze=True
):
    model = GPT2Model.from_pretrained(
        weight,
        cache_dir=cache_dir,
    )
    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False
    return model


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
