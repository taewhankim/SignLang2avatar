from seq2seq_translator.core.utils import load_yaml
from seq2seq_translator.core.sign_models.models import get_model_from_conf
from seq2seq_translator.core.utils.tokenizers import get_tokenizer_from_conf, preproc
from glob import glob
from pathlib import Path
from jamo import h2j, j2hcj
import torch, re

def to_jamo(text):
            return [jm for jm in j2hcj(h2j(text))]

class Translator:
    def __init__(self, model_conf='/mnt/workdir/sign/trains/0830_focal_loss/seq2seq_BertEncoder.yaml', \
                weight="/mnt/workdir/sign/infer_test/50_epochs.pth",
                tokenizer_conf="/mnt/workdir/sign/infer_test/tokenizer.yaml", device='cuda'):
        model_conf = load_yaml(model_conf)
        tok_conf = load_yaml(tokenizer_conf)
        self.model = get_model_from_conf(model_conf).eval().to(device)
        self.model.load_state_dict(torch.load(weight))
        self.encoder_tokenizer, self.decoder_tokenizer = get_tokenizer_from_conf(tok_conf)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def inference(self, input_sentence, beam=False):
        input_token = self.encoder_tokenizer.encode([input_sentence])
        if beam:
            result = self.model.infer_Beam(input_token, len(input_token.input_ids[0])+10, device=self.device ,k=3)
        else:
            result = self.model.infer_greedy(input_token, len(input_token.input_ids[0])+10, device=self.device)#,k=3)
        result = self.decoder_tokenizer.decode(result.tolist())[0]
        result = result.replace("[CLS]","").replace("[UNK]","").replace("[SEP]","").replace("[PAD]","")
        return result

class T2P:
    def __init__(self, folder_path = "/mnt/dms/HAND_SIGN/HS/all/"):
        self.wordlist = {}
        files = glob(folder_path+"/*.mp4")
        files += glob(folder_path+"/*.mkv")
        files += glob(folder_path+"/*.npz")
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
    def __init__(self, model_conf='/home/hand_sign_weight/seq2seq_BertEncoder.yaml', \
                weight="/home/hand_sign_weight/best_83epochs.pth",
                tokenizer_conf="/home/hand_sign_weight/tokenizer.yaml", 
                handsign_folder_path = "/mnt/dms/HAND_SIGN/val_demo/", device='cuda'):

        self.model = Translator(model_conf, weight, tokenizer_conf)        
        self.H2FPClass=T2P(handsign_folder_path)
    def translate(self, text):
        t = preproc(text)
        t = self.model.inference(t)
        if text[-1] == ".":
            t += " ."
        result = self.H2FPClass.t2p(t)
        return t, result, len(result)


if __name__ == "__main__":
    SF = Speech2HandsignFP()
    print(SF.translate("ㄴㅁㅇㄹㄴㅁㄹㅇ"))
