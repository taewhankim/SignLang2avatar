# environment

* cuda 11.2
* pytorch 1.9.1

# Setting 
* config
```
 ./main/config.py
    * weights & video folder & handchange_json *  

    hand_track_model = '/mnt/dms/KTW/hand4whole/weights/model_pretrain.pth'
    hand_model_path = '/mnt/dms/KTW/hand4whole/weights/snapshot_12_hand.pth.tar'
    face_model_path = '/mnt/dms/KTW/hand4whole/weights/snapshot_12_face.pth.tar'

    hand_sign_model_conf = "/mnt/dms/KTW/hand_sqs/weights2/seq2seq_BertEncoder.yaml"
    hand_sign_tokenizer_conf = "/mnt/dms/KTW/hand_sqs/weights2/tokenizer.yaml"
    hand_sign_weight_path = "/mnt/dms/KTW/hand_sqs/weights2/best_83epochs.pth"

    handsign_folder_path = "/mnt/dms/HAND_SIGN/val_npz/json"
    hand_change_list = "/mnt/dms/KTW/hand4whole/weights/hand_change_list.json"  ## 손 앞뒤 바뀌는 단어 목록

```
# Installation

* step 1 - setup
    `
    pip install -r requirements.txt
    `
* step 2 - detectron2

    * check version [detectron2_cuda_torch](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
    * install detectron2
    
* step 3
    `
    sudo apt install default-jdk
    `
# implement

## 1. only convert to video(Multiprocessing X , npz info X)
Run inference

```
python demo/hand/demo_hand.py --gpu 0 --input_text "나만의 탈 직접 만들어보기" --savefolder /mnt/dms/KTW/hand4whole/results/때2 --cont_n 15
```

* --gpu : set your gpu
* --input_text : input text (ex - "나만의 탈 직접 만들어보기" or ["나만의 탈 직접 만들어보기","속초는 좋다"])
* --savefolder : save directory
* --cont_n : stack frame num
* (optional) --rotation : rotate 270 degree

## 2. Multiprocessing with npz info
Run inference

```
python demo/hand/demo_hand_json_multi.py --gpu 0 --input_text "나만의 탈 직접 만들어보기" --savefolder /mnt/dms/KTW/hand4whole/results/때2 --cont_n 15
```

* --gpu : set your gpu
* --input_text : input text
* --savefolder : save directory
* --cont_n : stack frame num
* (optional) --rotation : rotate 270 degree

----
----
