import re
from jamo import h2j, j2hcj
from glob import glob
from pathlib import Path
# import readline
import sys, os
# from tkinter import W
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),"seq2seqTransformer"))

from seq2seqTransformer.core.data_classes import *
from seq2seqTransformer.core.TFmodel import *

def to_jamo(text):
    return [jm for jm in j2hcj(h2j(text))]

def load_transformer(vocab_path = "/mnt/dms/KTW/hand_sqs/weights/suwha_vocab",
                    ckpt_path = "/mnt/dms/KTW/hand_sqs/weights/output_1500epochs.ckpt"):
    vocab = suwhaVocabClass().load_spm(vocab_path)
    conf = TF_conf(src_vocab=vocab,tgt_vocab=vocab, ffn_hid_dim = 2048, nhead = 16, emb_size = 512)

    transformer = conf.load_Seq2SeqTF_from_conf()
    transformer.load_state_dict(torch.load(ckpt_path))
    transformer = transformer.eval()
    return transformer

def translate(model:Seq2SeqTransformer, input:str):
    return model.translate_beam(input)

class T2P:
    def __init__(self, folder_path = "/mnt/dms/HAND_SIGN/HS/all/"):
        self.wordlist = {}
        files = glob(folder_path+"*.mp4")
        for file in files:
            fp = Path(file)
            fn = fp.name.split(".")[0]
            self.wordlist[fn] = file

    def t2p(self, text):
        month_pattern = "[1]? [0-9] 월 "
        sub_months = re.findall(month_pattern,text)
        for m in sub_months:
            text = text.replace(m, m.replace(" ","")+" ")
        textl = text.split(" ")
        result = []
        for t in textl:
            if t in self.wordlist.keys():
                result.append(self.wordlist[t])
            else:
                jms = to_jamo(t)
                for jm in jms:
                    if jm in self.wordlist.keys():
                        result.append(self.wordlist[jm])
        return result

class Speech2HandsignFP:
    def __init__(self, vocab_path = "/mnt/dms/KTW/hand_sqs/weights/suwha_vocab",
                ckpt_path = "/mnt/dms/KTW/hand_sqs/weights/output_1500epochs.ckpt",
                handsign_folder_path = "/mnt/dms/HAND_SIGN/HS/all/"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.model = load_transformer(vocab_path, ckpt_path)        
        self.H2FPClass=T2P(handsign_folder_path)
    
    def translate(self, text):
        t = preproc(text)
        t = " ".join(translate(self.model, t))
        result = self.H2FPClass.t2p(t)
        return t, result, len(result)

# if
# shf = Speech2HandsignFP()
# # for i in range(1000):
# print(shf.translate("올레길 1 0 코스인 해안절경이 아름다운 송악산 산책"))
