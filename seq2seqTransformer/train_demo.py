from core.data_classes import *
from core.TFmodel import TF_conf
from core.train_utils import training

max_vocap_size = 1200
output_path = "/mnt/storage/HS/backup_track/suhwa/trains/0629_2"
if not os.path.exists(output_path):
    os.mkdir(output_path)
train_fp = "/mnt/storage/HS/backup_track/suhwa/seq2seqTransformer/datas/0629/0629"
valid_fp = "/mnt/storage/HS/backup_track/suhwa/seq2seqTransformer/datas/검증"
finetune = None
vocab = suwhaVocabClass(max_vocab_size=max_vocap_size).\
                        make_sentencepiece(input_file = train_fp, 
                        output_folder=output_path)

train_iter = suwhaDataset_from_text(train_fp, vocab).make_dataloader(64)
valid_iter = suwhaDataset_from_text(valid_fp, vocab).make_dataloader(64)


transformer = TF_conf(src_vocab=vocab,tgt_vocab=vocab, num_encoder_layers=12, ffn_hid_dim=256).load_Seq2SeqTF_from_conf()
if finetune:
    transformer.load_state_dict(torch.load(finetune))
    print("="*40)
    print("strat fine tune")
    print("="*40)
training(transformer, 1501, train_iter, valid_iter, output_root=output_path, output_name="output", save_iter=50)