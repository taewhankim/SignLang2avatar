from konlpy.tag import Mecab 
from .train_utils import Config
from ..models import KobertCRF
import pickle
import json 
import torch 

def get_pos_tagger(conf):
    if conf.name != "none":
        assert conf.name in ["Mecab"], "we only can use pos tagger: Mecab"
        if conf.name == "Mecab":
            return Mecab(conf.user_dic_dir)
    else:
        return None 

class PosTagger():
    def __init__(self, conf):
        self.pos_tagger_name = conf.name
        self.tagger = get_pos_tagger(conf)
        if self.pos_tagger_name == "Mecab":
            self.pos_to_idx = {list(self.tagger.tagset.keys())[i]: i for i in range(len(self.tagger.tagset))}
            
            # 없는 token 
            self.pos_to_idx['UNA'] = 43
            self.pos_to_idx['SPACE'] = 44 
            self.pos_to_idx['START'] = 45
            self.pos_to_idx['END'] = 46
            self.pos_to_idx['PAD'] = 47
            self.pos_to_idx['UNKNOWN'] = len(self.pos_to_idx)

    def tagging(self, sentence):
        if self.pos_tagger_name == "Mecab":
            tagged_sentence = self.tagger.pos(sentence)
            tagged_sentence_idx = list()
            for word, pos in tagged_sentence:
                one_tagged_sentence_idx = list()
                pos_split = pos.split('+')
                for one_pos in pos_split:
                    one_tagged_sentence_idx.append(self.pos_to_idx[one_pos])
                tagged_sentence_idx.append(one_tagged_sentence_idx)
            return tagged_sentence, tagged_sentence_idx

    def batch_tagging(self, batch_sentence):
        batch_tokenized_sentence = list()
        batch_tokenized_sentence_idx = list()
        for sentence in batch_sentence:
            tagged_sentence, tagged_sentence_idx = self.tagging(sentence)
            batch_tokenized_sentence.append(tagged_sentence)
            batch_tokenized_sentence_idx.append(tagged_sentence_idx)
        return batch_tokenized_sentence, batch_tokenized_sentence_idx


def get_ner_tagger(conf, len_ner_idx = None):
    model_config = Config(json_path = conf.config_loc)
    with open(conf.vocab_loc, 'rb') as f:
        vocab = pickle.load(f)
    model = KobertCRF(config = model_config, num_classes = len_ner_idx, vocab = vocab)
    model_dict = model.state_dict()
    checkpoint = torch.load(conf.checkpoint_loc, map_location=torch.device('cpu'))
    convert_keys = {}
    
    # checkpoint['model_state_dict']['crf.trans_matrix'] = checkpoint['model_state_dict'].pop('crf.transitions')
    # checkpoint['model_state_dict']['crf.start_trans'] = checkpoint['model_state_dict'].pop('crf.start_transitions')
    # checkpoint['model_state_dict']['crf.end_trans'] = checkpoint['model_state_dict'].pop('crf.end_transitions')
    
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not in the model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v
    model.load_state_dict(convert_keys)
    return model 


class NERTagger():
    def __init__(self, conf):
        self.ner_tagger_name = conf.name
        with open(conf.ner_to_idx_loc) as f:
            self.ner_to_idx = json.load(f)
            self.idx_to_ner = {v: k for k,v in self.ner_to_idx.items()}
        self.tagger = get_ner_tagger(conf, len_ner_idx = len(self.idx_to_ner))

#     def tagging()

