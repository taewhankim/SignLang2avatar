import torch
import sentencepiece as spm
import sys, os
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from konlpy.tag import Twitter

twt = Twitter()

import re

import Levenshtein as Lev

def wer(ref, hyp ,debug=False):
    r = ref.split()
    h = hyp.split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY=1 # Tact
    INS_PENALTY=1 # Tact
    SUB_PENALTY=1 # Tact
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("Ncor " + str(numCor))
        print("Nsub " + str(numSub))
        print("Ndel " + str(numDel))
        print("Nins " + str(numIns))
    return (numSub + numDel + numIns) / (float)(len(r)) # numCor, numSub, numDel, numIns, 
    
# Run:
# ref='Tuan anh mot ha chin'
# hyp='tuan anh mot hai ba bon chin'
# wer(ref, hyp ,debug=True)

def cer(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')
    dist = Lev.distance(hyp, ref)
    length = len(ref)
    return dist/length # dist, length, 

def preproc(line, is_nl=False):
    # pre1 괄호제거
    pre1 = "[\(\[\<\]\)][ㄱ-ㅎ가-힣0-9a-zA-Z,. \|\n?!:]+[\(\[\<\>\]\)]"
    # pre2 특문제거
    pre2 = "[^ㄱ-ㅎ가-힣0-9a-zA-Z,. \|\n?!:]"
    # pre34 대문자, 숫자 떨어뜨리기 + 앞 뒤 스페이스
    pre3 = "([A-Z][^a-zA-Z ]|[^a-zA-Z ][A-Z])"
    pre4 = "[A-Z]{2,100}|[0-9]+"
    # pre5 = 영단어 앞, 뒤 스페이스
    pre5 = "[A-Z][a-z]+|[a-z]+|[.,!?~:]+|[0-9]+"
    # pre6 = 스페이스 2회 이상 -> 1회
    pre6 = "[ ]{2,}"

    l = line.strip().strip("\n")
    l = re.sub(pre1, "",l)
    l = re.sub(pre2, " ", l)
    uppers = re.findall(pre3,l)
    for upp in uppers: 
        re_case = " "+" ".join([u for u in upp])+" "
        l = l.replace(upp, re_case)
    uppers = re.findall(pre4,l)
    for upp in uppers: 
        re_case = " "+" ".join([u for u in upp])+" "
        l = l.replace(upp, re_case)
    words = re.findall(pre5, l)

    # 형태소 분석에 의한 split
    if is_nl:
        tags = twt.pos(l)
        l = " ".join([i for i, j in tags]) 

    for w in words:
        l = l.replace(w, " "+w+" ")

    spaces = re.findall(pre6, l)
    for s in spaces:
        l = l.replace(s, " ")

    spaces = re.findall(pre6, l)
    for s in spaces:
        l = l.replace(s, " ")    

    return l.strip()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class suwhaVocabClass():
    def __init__(self, max_vocab_size=2000):
        self.vocab_size = None
        self.vocab_path = None
        self.num_vocab_dict = None
        self.vocab_num_dict = None
        self.max_vocab_size = max_vocab_size
        self.spm = spm.SentencePieceProcessor()
        self.USER_TOKENS = {"[PAD]":0, "[UNK]":1, "[BOS]":2, "[EOS]":3, "[SEP]":4, "[CLS]":5, "[MASK]":6}

    def load_spm(self, vocab_path=None):
        if vocab_path == None:
            vocab_path = self.vocab_path
        self.spm.load(vocab_path+".model")
        with open(vocab_path+".vocab", 'r', encoding="utf-8") as f:
            lines = f.readlines()
        self.vocab_size = len(self.spm)
        self.vocab_path = vocab_path
        self.num_vocab_dict = {i+1:text.split()[0].strip() for i, text in enumerate(lines)}
        self.vocab_num_dict = dict(zip(self.num_vocab_dict.values(), self.num_vocab_dict.keys()))
        return self

    def make_sentencepiece(self, input_file, output_folder='.', output_name='suwha_vocab'):
        sp_input_file = output_folder+"/sptemp.txt"
        output_folder = output_folder+"/vocab"
        with open(input_file, "r") as inf:
            with open(sp_input_file, "w") as spf:
                lines = inf.readlines()
                for line in lines:
                    try:
                        ta, tr = line.strip().strip("\n").split("|")
                    except:
                        print(line.strip().strip("\n").split("|"))
                        exit()
                    ta = preproc(ta)
                    tr = preproc(tr,True)
                    write_line = ta+"|"+tr+"\n"
                    if write_line != "|\n":
                        spf.write(write_line)

        vocab_size = self.max_vocab_size
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        vocap_path = os.path.abspath(os.path.join(output_folder, output_name))
        self.vocab_size = vocab_size
        spm.SentencePieceTrainer.train(
            f"--input={sp_input_file} --model_prefix={vocap_path} --vocab_size={vocab_size + 7}" + 
            " --model_type=bpe" +
            " --max_sentence_length=999999" + # 문장 최대 길이
            " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
            " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
            " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
            " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
            " --user_defined_symbols=[SEP],[CLS],[MASK]") # 사용자 정의 토큰
        self.vocab_path = vocap_path
        return self.load_spm(self.vocab_path)
    
    def __len__(self):
        return self.vocab_size+7

    def to_subword(self,data):
        return self.spm.encode_as_pieces(data)

    def __call__(self, data):
        self.spm.encode_as_pieces(data)
        return self.vocab_to_num(data)

    def vocab_to_num(self, data):
        try:
            temp = [self.vocab_num_dict[i] if i in self.vocab_num_dict.keys() else self.vocab_num_dict['[UNK]'] for i in data]
        except:
            temp = [self.vocab_num_dict[i] if i in self.vocab_num_dict.keys() else self.vocab_num_dict['<unk>'] for i in data]

        return temp
    def num_to_vocab(self, data):
        return [self.num_vocab_dict[i] for i in data]

class suwhaDataset_from_text(Dataset):
    def __init__(self, fp=None, vocab=None):
        '''fp= train or test or valid file path'''
        '''vocab = vocab class'''
        super().__init__()
        self.vocab = vocab # type:suwhaVocabClass
        self.datas = []
        with open(fp, 'r') as f:
            lines = f.readlines()
        for line in lines:
            try:
                t, s = line.strip().strip("\n").split('|')
                t, s = preproc(t), preproc(s)
                self.datas.append([self.vocab(s), self.vocab(t)])
            except:
                pass

        self.usr_token_dict = self.vocab.USER_TOKENS
        self.BOS = self.usr_token_dict["[BOS]"]
        self.EOS = self.usr_token_dict["[EOS]"]
        self.PAD = self.usr_token_dict["[PAD]"]
        # maxlen = max([len(i) for i in (self.source+self.target)])
        # self.source = np.array([np.pad(self.vocab.vocab_to_num(i),(0,maxlen-len(i)),constant_values=0) for i in self.source])
        # self.target = np.array([np.pad(self.vocab.vocab_to_num(i),(maxlen-len(i),0), constant_values=0) for i in self.target])

    def __getitem__(self, index):
        return self.source[index], self.target[index]

    def __len__(self):
        return len(self.source)

    def generate_batch(self, datas):
        src, target = [], []
        for (src_item, tgt_item) in datas:
            src.append(torch.cat([torch.tensor([self.BOS]), torch.tensor(src_item), torch.tensor([self.EOS])], dim=0))
            target.append(torch.cat([torch.tensor([self.BOS]), torch.tensor(tgt_item), torch.tensor([self.EOS])], dim=0))
        src = pad_sequence(src, padding_value=self.PAD)
        target = pad_sequence(target, padding_value=self.PAD)
        return src, target

    def make_dataloader(self, bsz):
        iter = DataLoader(self.datas, batch_size=bsz,
                            shuffle=True, collate_fn=self.generate_batch)
        return iter
