# coding=utf-8
# Copyright 2019 SK T-Brain Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ctypes import Union
import os, sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cur_path, ".."))
from zipfile import ZipFile
import torch
from transformers import BertModel

from .kobert_vocab import BertVocabClass
from kobert import download, get_tokenizer


def get_pytorch_kobert_model(cachedir=".cache"):
    pytorch_kobert = {
        "url": "s3://skt-lsl-nlp-model/KoBERT/models/kobert_v1.zip",
        "chksum": "411b242919",  # 411b2429199bc04558576acdcac6d498
    }

    # download model
    model_info = pytorch_kobert
    model_path, is_cached = download(
        model_info["url"], model_info["chksum"], cachedir=cachedir
    )
    cachedir_full = os.path.expanduser(cachedir)
    zipf = ZipFile(os.path.expanduser(model_path))
    zipf.extractall(path=cachedir_full)
    model_path = os.path.join(os.path.expanduser(cachedir), "kobert_from_pretrained")
    # download vocab
    vocab_path = get_tokenizer()
    return get_pretrained_kobert(model_path, vocab_path)


def get_pretrained_kobert(cachedir=".cache", device='cuda'):
    pytorch_kobert = {
        "url": "s3://skt-lsl-nlp-model/KoBERT/models/kobert_v1.zip",
        "chksum": "411b242919",  # 411b2429199bc04558576acdcac6d498
    }

    # download model
    model_info = pytorch_kobert
    model_path, is_cached = download(
        model_info["url"], model_info["chksum"], cachedir=cachedir
    )
    cachedir_full = os.path.expanduser(cachedir)
    zipf = ZipFile(os.path.expanduser(model_path))
    zipf.extractall(path=cachedir_full)
    model_path = os.path.join(os.path.expanduser(cachedir), "kobert_from_pretrained")
    # download vocab
    bertmodel = BertModel.from_pretrained(
        model_path, return_dict=False).to(device)
    return bertmodel


# def get_pretrained_kobert(
#     pret_path=cur_path + "/pretrained", device="cuda"
# ) -> BertModel:
#     model_path = pret_path + "/kobert_from_pretrained"
#     bertmodel = BertModel.from_pretrained(
#         model_path, return_dict=False
#     )  # type: BertModel

#     device = torch.device(device)
#     bertmodel.to(device)
#     return bertmodel


def get_kobert_vocab(pret_path=cur_path + "/pretrained") -> BertVocabClass:
    vocab_file = pret_path + "/vocab"
    vocab_b_obj = BertVocabClass(vocab_file=vocab_file)
    return vocab_b_obj


def get_empty_kobert(pret_path=cur_path + "/pretrained", device="cuda"):
    bertmodel = get_pretrained_kobert(pret_path=pret_path, device=device)
    bertmodel.init_weights()
    return bertmodel
