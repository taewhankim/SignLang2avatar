from math import log
from typing import List
import torch
from .custom_layers import *
from .loss import *
from seq2seq_translator.core.utils.tokenizers import InputToken, preproc
from torch import nn
from einops import rearrange
from munch import Munch

# from seq2seq_translator.core import str2code, get_layers_from_conf
import sys

CUDA_LAUNCH_BLOCKING = 1


def generate_square_subsequent_mask(sz: int, device="cuda"):
    return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)


class flow:
    def __init__(self, token, score=0, len=1):
        self.score = score
        self.len = len
        self.tokens = token

    @property
    def panalty(self, alpha=1.2, min_length=3):
        return ((min_length + self.len) / (min_length + 1)) ** alpha

    @property
    def beamscore(self):
        return self.score  # /self.panalty

    def __lt__(self, other):
        return self.beamscore < other.beamscore

    def __eq__(self, other):
        return self.beamscore == other.beamscore

    def __str__(self):
        return f"score:{self.score}, len:{self.len}, tokens:{self.tokens}, beamscore:{self.beamscore}"

class Seq2SeqBertEncoder(nn.Module):
    def __init__(self, conf):
        super(Seq2SeqBertEncoder, self).__init__()
        if "model" in conf.keys():
            conf = conf.model

        layers = get_layers_from_conf(conf)

        if conf.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.de_emb = layers["emb_decoder"]
        self.encoder = layers["encoder"]
        self.decoder = layers["decoder"]
        self.loss = layers["loss"]
        self.generator = layers["generator"]
        self.conf = conf

    def forward(
        self, tokenized_korean: InputToken, tokenized_sign: InputToken, device="cuda"
    ):

        tgt_emb = self.de_emb(tokenized_sign.tgt_token.input_ids.transpose(0, 1))

        with torch.no_grad():
            memory = self.encoder(
                tokenized_korean.input_ids,
                attention_mask=tokenized_korean.attention_mask,
            )[0]

        tgt_mask = generate_square_subsequent_mask(
            max(tokenized_sign.input_len) - 1, device=device
        )

        # outs : seq_len * batch * emb
        outs = self.decoder(
            tgt_emb,
            memory.transpose(0, 1),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(tokenized_sign.tgt_token.attention_mask == 0),
            memory_key_padding_mask=tokenized_korean.attention_mask == 0,
        )

        preds = self.generator(outs)
        # preds : batch * seq_len * vocab_size
        tgt_out = tokenized_sign.input_ids[:, 1:].transpose(0, 1)
        loss = self.loss(preds.reshape(-1, preds.shape[-1]), tgt_out.reshape(-1))

        return loss

    def infer_greedy(
        self, input_ids, max_len=128, device="cuda"
    ):  # batch x 1, decoder input 입력으로 알고싶은 단어의 index
        """beam_search=False or k"""
        input_ids = input_ids.input_ids

        tgt = torch.tensor([[2]]).to(device)

        with torch.no_grad():
            memory = self.encoder(input_ids)[0]

        next_word = None

        while tgt.shape[1] < max_len and next_word != 3:  # pred != [[3]]:
            tgt_emb = self.de_emb(tgt.transpose(1, 0))
            tgt_mask = generate_square_subsequent_mask(
                sz=tgt_emb.shape[0], device=device
            ).type(torch.bool)
            out = self.decoder(tgt_emb, memory.transpose(0, 1), tgt_mask=tgt_mask)[
                -1
            ].unsqueeze(1)

            preds = self.generator(out)[0]
            _, next_word = torch.max(preds, dim=1)
            tgt = torch.cat([tgt, torch.tensor([[next_word]]).to(device)], dim=1)

        return tgt

    def infer_beam_greedy(
        self,
        input_token,
        max_len=128,
        device="cuda",
        k=5,
        top_p=0.92,
    ):  # batch x 1, decoder input 입력으로 알고싶은 단어의 index
        """beam_search=False or k"""

        def decode_step(tgt, memory, k=5, top_p=0.92):
            tgt_emb = self.de_emb(tgt.transpose(1, 0))
            tgt_mask = generate_square_subsequent_mask(
                sz=tgt_emb.shape[0], device=device
            ).type(torch.bool)
            outs = self.decoder(tgt_emb, memory.transpose(0, 1), tgt_mask=tgt_mask)
            out = outs.transpose(0, 1)
            preds = self.generator(out[:, -1])
            # means = preds.mean(dim=-1, keepdim=True)
            # stds = preds.std(dim=-1, keepdim=True)
            # preds = (preds - means) / stds
            scores, preds = torch.topk(preds.softmax(dim=-1), k, -1)
            scores, preds = scores[0], preds[0]
            # p-sampling
            score_sum = 0

            idx = 0
            for i in range(len(scores)):
                idx = i + 1
                if score_sum + scores[i] > top_p:
                    break
                score_sum += scores[i]

            scores = scores[:idx]
            preds = preds[:idx]
            return scores, preds

        input_ids = input_token.input_ids

        tg = torch.tensor([[2]]).to(device)

        with torch.no_grad():
            mem = self.encoder(input_ids)[0]

        score, preds = decode_step(tg, mem, k, top_p=top_p)
        cur_q = [
            flow(
                torch.cat((tg, p.reshape(1, -1)), 1),
                score=1 * (-1 * log(s + 1e-10)),
                len=1,
            )
            for s, p in zip(score, preds)
        ]
        tempk = k
        result = []
        while len(cur_q) > 0 and tempk > 0:
            temp = []  # type:List[flow]
            for tgt in cur_q:
                tg = tgt.tokens
                le = tgt.len
                sc = tgt.score
                scores, preds = decode_step(tg, mem, tempk)
                flows = [
                    flow(
                        torch.cat((tg, p.reshape(1, -1)), 1),
                        score=sc * (-1 * log(s + 1e-8)),
                        len=le + 1,
                    )
                    for s, p in zip(scores, preds)
                ]
                flows
                [temp.append(f) for f in flows]

            cur_q = []
            topk = sorted(temp)[:k]
            [
                result.append(t)
                if t.tokens[0][-1] == 3 or len(t.tokens[0]) > max_len
                else cur_q.append(t)
                for t in topk
            ]
            tempk = k - len(result)
        # print([r.score for r in result])
        Beam = sorted(result)[0].tokens
        greedy = self.infer_greedy(input_token, device=device)

        return Beam, greedy

    def infer_Beam(
        self,
        input_token,
        max_len=128,
        device="cuda",
        k=5,
        top_p=0.92,
    ):  # batch x 1, decoder input 입력으로 알고싶은 단어의 index
        """beam_search=False or k"""

        def decode_step(tgt, memory, k=5, top_p=0.92):
            tgt_emb = self.de_emb(tgt.transpose(1, 0))
            tgt_mask = generate_square_subsequent_mask(
                sz=tgt_emb.shape[0], device=device
            ).type(torch.bool)
            outs = self.decoder(tgt_emb, memory.transpose(0, 1), tgt_mask=tgt_mask)
            out = outs.transpose(0, 1)
            preds = self.generator(out[:, -1])
            # means = preds.mean(dim=-1, keepdim=True)
            # stds = preds.std(dim=-1, keepdim=True)
            # preds = (preds - means) / stds
            scores, preds = torch.topk(preds.softmax(dim=-1), k, -1)
            scores, preds = scores[0], preds[0]
            # p-sampling
            score_sum = 0

            idx = 0
            for i in range(len(scores)):
                idx = i + 1
                if score_sum + scores[i] > top_p:
                    break
                score_sum += scores[i]

            scores = scores[:idx]
            preds = preds[:idx]
            return scores, preds

        input_ids = input_token.input_ids

        tg = torch.tensor([[2]]).to(device)

        with torch.no_grad():
            mem = self.encoder(input_ids)[0]

        score, preds = decode_step(tg, mem, k, top_p=top_p)
        cur_q = [
            flow(
                torch.cat((tg, p.reshape(1, -1)), 1),
                score=1 * (-1 * log(s + 1e-10)),
                len=1,
            )
            for s, p in zip(score, preds)
        ]
        tempk = k
        result = []
        while len(cur_q) > 0 and tempk > 0:
            temp = []  # type:List[flow]
            for tgt in cur_q:
                tg = tgt.tokens
                le = tgt.len
                sc = tgt.score
                scores, preds = decode_step(tg, mem, tempk)
                flows = [
                    flow(
                        torch.cat((tg, p.reshape(1, -1)), 1),
                        score=sc * (-1 * log(s + 1e-8)),
                        len=le + 1,
                    )
                    for s, p in zip(scores, preds)
                ]
                [temp.append(f) for f in flows]

            cur_q = []
            topk = sorted(temp)[:k]
            [
                result.append(t)
                if t.tokens[0][-1] == 3 or len(t.tokens[0]) > max_len
                else cur_q.append(t)
                for t in topk
            ]
            tempk = k - len(result)

        Beam = sorted(result)[0].tokens
        return Beam  # , greedy

    def preproc(self, input_str):
        return preproc(input_str)

    def trainable_layers(self):
        total = [
            self.de_emb,
            self.encoder,
            self.decoder,
            self.generator,
            self.loss,
        ]
        results = [
                layer
                for layer in total
                if layer != None and layer != False and check_grad(layer)
            ]
        return results


def get_model_from_conf(conf):
    return str2code(conf.model.base)(conf.model)


def get_base_from_conf(conf):

    if "emb" in conf:
        emb_conf = conf.emb

        if "encoder" in emb_conf:
            en_emb_conf = emb_conf.encoder
            if en_emb_conf.name != "none":
                en_emb = str2code(en_emb_conf.name)(**en_emb_conf.properties)
            else:
                en_emb = None
        else:
            en_emb = None

        if "decoder" in emb_conf:
            de_emb_conf = emb_conf.decoder
            if de_emb_conf.name != "none":
                de_emb = str2code(de_emb_conf.name)(**de_emb_conf.properties)
            else:
                de_emb = None
        else:
            de_emb = None

        if "emb_pos" in conf:
            en_emb_pos_conf = conf.emb_pos
            if en_emb_pos_conf.name != "none":
                en_pos_emb = str2code(en_emb_pos_conf.name)(
                    **en_emb_pos_conf.properties
                )
            else:
                en_pos_emb = None
        else:
            en_pos_emb = None

        if "emb_ner" in conf:
            en_emb_ner_conf = conf.emb_ner
            if en_emb_ner_conf.name != "none":
                en_ner_emb = str2code(en_emb_ner_conf.name)(
                    **en_emb_ner_conf.properties
                )
            else:
                en_ner_emb = None
        else:
            en_ner_emb = None

    else:
        en_emb = None
        de_emb = None
        en_pos_emb = None
        en_ner_emb = None

    if "encoder" in conf:
        en_conf = conf.encoder
        if en_conf.name != "none":
            encoder = str2code(en_conf.name)(**en_conf.properties)
        else:
            encoder = None
    else:
        encoder = None

    if "decoder" in conf:
        de_conf = conf.decoder
        if de_conf.name != "none":
            decoder = str2code(de_conf.name)(**de_conf.properties)
        else:
            decoder = None
    else:
        decoder = None

    if "loss" in conf:
        loss_conf = conf.loss
        if loss_conf.name != "none":
            loss = str2code(loss_conf.name)()
        else:
            loss = None
    else:
        loss = CrossEntropyLoss()

    return {
        "en_emb": en_emb,
        "en_pos_emb": en_pos_emb,
        "en_ner_emb": en_ner_emb,
        "de_emb": de_emb,
        "encoder": encoder,
        "decoder": decoder,
        "loss": loss,
    }


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def str2code(classname):
    return getattr(sys.modules[__name__], classname)


def getlayer(conf: Munch, key=""):
    deep = {}
    key = key.strip("_")
    if "name" in conf.keys() and conf.name != "none":
        if "properties" in conf:
            return {key: str2code(conf.name)(**conf.properties)}
        else:
            return {key: str2code(conf.name)()}
    else:
        for k, v in conf.items():
            if type(v) == Munch:
                deep.update(getlayer(conf[k], key=key + "_" + k))

    return deep


def get_layers_from_conf(conf: Munch):

    if type(conf) == dict:
        conf = Munch.fromDict(conf)
    if "model" in conf.keys():
        c = conf.model
    else:
        c = conf
    layers = getlayer(c)
    return layers


def check_grad(model: Module):
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            return False
    return True
