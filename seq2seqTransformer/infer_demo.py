from core.data_classes import *
from core.TFmodel import *

import os

if __name__ == "__main__":
    # os.remove("./trained/TF*")
    vocab = suwhaVocabClass().load_spm("/mnt/storage/HS/backup_track/suhwa/vocab/suwha_vocab")
    conf = TF_conf(vocab, vocab)
    transformer = conf.load_Seq2SeqTF_from_conf()
    transformer.load_state_dict(torch.load("/mnt/storage/HS/backup_track/suhwa/output_0614_min_val.ckpt"))
    transformer = transformer.eval()

    while True:
        strr=input(">>>")
        print(transformer.translate_beam(strr))
        # print(transformer.translate_greedy(strr))
        