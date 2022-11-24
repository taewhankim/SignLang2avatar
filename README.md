# environment

* cuda 11.2
* pytorch 1.9.1
* weights 
```
    hand_track_model = '/mnt/dms/KTW/hand4whole/weights/model_pretrain.pth'
    hand_model_path = '/mnt/dms/KTW/hand4whole/weights/snapshot_12_hand.pth.tar'
    face_model_path = '/mnt/dms/KTW/hand4whole/weights/snapshot_12_face.pth.tar'

    hand_sign_model_conf = "/mnt/dms/KTW/hand_sqs/weights2/seq2seq_BertEncoder.yaml"
    hand_sign_tokenizer_conf = "/mnt/dms/KTW/hand_sqs/weights2/tokenizer.yaml"
    hand_sign_weight_path = "/mnt/dms/KTW/hand_sqs/weights2/best_83epochs.pth"

    handsign_folder_path = "/mnt/dms/HAND_SIGN/val_npz/json"

```
# Setting 
* config
```
 ./main/config.py
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

## 1. only video(Multiprocessing X , npz info X)
Run inference

```
python demo/hand/demo_hand.py --gpu 0 --input_text "나만의 탈 직접 만들어보기" --savefolder /mnt/dms/KTW/hand4whole/results/때2 --cont_n 15
```

* --gpu : set your gpu
* --input_text : input text
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

# **Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation (Pose2Pose codes)**
  
<p align="center">  
<img src="assets/qualitative_results.png">  
</p> 

<p align="middle">
<img src="assets/3DPW_1.gif" width="720" height="240"><img src="assets/3DPW_2.gif" width="720" height="240">
</p>

High-resolution video link: [here](https://youtu.be/Ym_CH8yxBso)


## Introduction  
This repo is official **[PyTorch](https://pytorch.org)** implementation of **[**Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation** (CVPRW 2022)](https://arxiv.org/abs/2011.11534)**. **This repo contains body-only, hand-only, and face-only codes of the Pose2Pose. The whole-body codes of the Hand4Whole are available at [here](https://github.com/mks0601/Hand4Whole_RELEASE).**
  
  
## Quick demo  
* Slightly change `torchgeometry` kernel code following [here](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527).
* Download the pre-trained Pose2Pose from any of [here (body)](https://drive.google.com/file/d/1TY8R4sgM9p05MAtXGs3SjDvQQU3wINxK/view?usp=sharing), [here (hand)](https://drive.google.com/file/d/18vLbJSr0FaTpzqPYdCNHDmhXbE5yeeOJ/view?usp=sharing), and [here (face)](https://drive.google.com/file/d/1LFhdpvKVtrEK6kzI0NTalx99tv64q5_Z/view?usp=sharing).
* Prepare `input.png` and pre-trained snapshot at any of `demo/body`, `demo/hand`, and `demo/face` folders.
* Download [human_model_files](https://drive.google.com/drive/folders/1jOzMo9Rl0iSgbzGiYBKlxuEKwmCih1qc?usp=sharing) and it at `common/utils/human_model_files`.
* Go to any of `demo/body`, `demo/hand`, and `demo/face` folders and edit `bbox`.
* Run `python demo.py --gpu 0`.
* If you run this code in ssh environment without display device, do follow:
```
1、Install oemesa follow https://pyrender.readthedocs.io/en/latest/install/
2、Reinstall the specific pyopengl fork: https://github.com/mmatl/pyopengl
3、Set opengl's backend to egl or osmesa via os.environ["PYOPENGL_PLATFORM"] = "egl"
```

## Directory  
### Root  
The `${ROOT}` is described as below.  
```  
${ROOT}  
|-- data  
|-- demo
|-- common  
|-- main  
|-- output  
```  
* `data` contains data loading codes and soft links to images and annotations directories.  
* `demo` contains demo codes.
* `common` contains kernel codes for Pose2Pose.  
* `main` contains high-level codes for training or testing the network.  
* `output` contains log, trained models, visualized outputs, and test result.  
  
### Data  
You need to follow directory structure of the `data` as below.  
```  
${ROOT}  
|-- data  
|   |-- AGORA
|   |   |-- data
|   |   |   |-- AGORA_train.json
|   |   |   |-- AGORA_validation.json
|   |   |   |-- AGORA_test_bbox.json
|   |   |   |-- 1280x720
|   |   |   |-- 3840x2160
|   |-- FFHQ
|   |   |-- FFHQ_FLAME_NeuralAnnot.json
|   |   |-- FFHQ.json
|   |-- FreiHAND
|   |   |-- data
|   |   |   |-- training
|   |   |   |-- evaluation
|   |   |   |-- freihand_train_coco.json
|   |   |   |-- freihand_train_data.json
|   |   |   |-- freihand_eval_coco.json
|   |   |   |-- freihand_eval_data.json
|   |-- Human36M  
|   |   |-- images  
|   |   |-- annotations   
|   |-- InterHand26M
|   |   |-- images
|   |   |-- annotations
|   |-- MPII
|   |   |-- data
|   |   |   |-- images
|   |   |   |-- annotations
|   |-- MPI_INF_3DHP
|   |   |-- data
|   |   |   |-- images_1k
|   |   |   |-- MPI-INF-3DHP_1k.json
|   |   |   |-- MPI-INF-3DHP_camera_1k.json
|   |   |   |-- MPI-INF-3DHP_joint_3d.json
|   |   |   |-- MPI-INF-3DHP_SMPL_NeuralAnnot.json
|   |-- MSCOCO  
|   |   |-- images  
|   |   |   |-- train2017  
|   |   |   |-- val2017  
|   |   |-- annotations 
|   |-- PW3D
|   |   |-- data
|   |   |   |-- 3DPW_train.json
|   |   |   |-- 3DPW_validation.json
|   |   |   |-- 3DPW_test.json
|   |   |-- imageFiles

```

* 3D body datasets: AGORA, Human3.6M, MPII, MPI-INF-3DHP, MSCOCO, and 3DPW
* 3D hand datasets: AGORA, FreiHAND, and MSCOCO
* 3D face datasets: AGORA, FFHQ, and MSCOCO
* Download AGORA parsed data [[data](https://drive.google.com/drive/folders/1ZaoYEON2WX9O_8gyPVsnBO2hph8v6lPS?usp=sharing)][[parsing codes](tool/AGORA)]
* Download FFHQ parsed data and FLAME parameters [[data](https://drive.google.com/file/d/1lG8rakysBXyzwNaTlmDD0hRtQQl8Dssw/view?usp=sharing)][[FLAME parameters from NeuralAnnot](https://drive.google.com/file/d/1MtEtal-mmE9j36f_Nz160E_N1CLK07yf/view?usp=sharing)]
* Download FreiHAND parsed data [[data](https://drive.google.com/drive/folders/1QGWu_nWi5eyrWSkPEOMyK1tAGFWRXQjC?usp=sharing)]
* Download Human3.6M parsed data and SMPL parameters [[data](https://drive.google.com/drive/folders/1kgVH-GugrLoc9XyvP6nRoaFpw3TmM5xK?usp=sharing)][[SMPL parameters from NeuralAnnot](https://drive.google.com/drive/folders/1xLkuyrjB832o5aG_M3g3EEf0PXqKkvS8?usp=sharing)]
* Download InterHand2.6M dataset from [here](https://mks0601.github.io/InterHand2.6M/).
* Download MPII parsed data and SMPL parameters [[data](https://drive.google.com/drive/folders/1MmQ2FRP0coxHGk0Ntj0JOGv9OxSNuCfK?usp=sharing)][[SMPL parameters from NeuralAnnot](https://drive.google.com/file/d/1dvtXmRWuTw1Rv89I8uGFl-YkZbhg3Lqz/view?usp=sharing)]
* Download MPI-INF-3DHP parsed data and SMPL parameters [[data](https://drive.google.com/drive/folders/1oHzb4oJHPZllLgN_yjyatp1LdqdP0R61?usp=sharing)][[SMPL parameters from NeuralAnnot](https://drive.google.com/file/d/1mxyPTnwM7D5L0NhtSEY1-pl3k5mS2IV6/view?usp=sharing)]
* Download MSCOCO SMPL parameters [[SMPL parameters from NeuralAnnot](https://drive.google.com/file/d/14XDSCdvpW_fJe_plbQ9wPwLv2VjdNuYZ/view?usp=sharing)]
* Download 3DPW parsed data [[data](https://drive.google.com/drive/folders/1fWrx0jnWzcudU6FN6QCZWefaOFSAadgR?usp=sharing)]
* All annotation files follow [MSCOCO format](http://cocodataset.org/#format-data). If you want to add your own dataset, you have to convert it to [MSCOCO format](http://cocodataset.org/#format-data).  
  
  
### Output  
You need to follow the directory structure of the `output` folder as below.  
```  
${ROOT}  
|-- output  
|   |-- log  
|   |-- model_dump  
|   |-- result  
|   |-- vis  
```  
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.  
* `log` folder contains training log file.  
* `model_dump` folder contains saved checkpoints for each epoch.  
* `result` folder contains final estimation files generated in the testing stage.  
* `vis` folder contains visualized results.  


## Running Pose2Pose
* In the `main/config.py`, you can change datasets to use.  

### Train 
In the `main` folder, run  
```bash  
python train.py --gpu 0-3 --parts body
```  
to train body-only Pose2Pose on the GPU 0,1,2,3. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`. You can chnage `body` to `hand` or `face` for the hand-only and face-only Pose2Pose, respectively. To train body-only Pose2Pose from the pre-trained 2D human pose estimation network, download [this](https://drive.google.com/file/d/1zHAVs1v0Ix03ug5Ym425YE3gKr8GpeAn/view?usp=sharing) and place it at `output/model_dump`. Then, run
```bash  
python train.py --gpu 0-3 --parts body --continue
```  

  
### Test  
Place trained model at the `output/model_dump/`. 
  
In the `main` folder, run  
```bash  
python test.py --gpu 0-3 --parts body --test_epoch 6
```  
to test body-only Pose2Pose on the GPU 0,1,2,3 with60th epoch trained model. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`.  You can chnage `body` to `hand` or `face` for the hand-only and face-only Pose2Pose, respectively.
  
## Models
* Download body-only Pose2Pose trained on H36M+MSCOCO+MPII from [here](https://drive.google.com/file/d/1TY8R4sgM9p05MAtXGs3SjDvQQU3wINxK/view?usp=sharing).
* Download body-only Pose2Pose fine-tuned on AGORA from [here](https://drive.google.com/file/d/1DjJcKUzEtYD9uMzvuh7ixgmaxTonP5vK/view?usp=sharing).
* Download hand-only Pose2Pose trained on FreiHAND+InterHand2.6M+MSCOCO from [here](https://drive.google.com/drive/folders/1zzCsRwGj58GNvxb9_yWk9GVPnHK5LP4s?usp=sharing).
* Download face-only Pose2Pose trained on FFHQ+MSCOCO from [here](https://drive.google.com/file/d/1LFhdpvKVtrEK6kzI0NTalx99tv64q5_Z/view?usp=sharing).

## Results

### 3D body-only and hand-only results
<p align="middle">
<img src="assets/AGORA_SMPL.PNG" width="450" height="300">
</p>


<p align="middle">
<img src="assets/3DPW.PNG" width="360" height="264">
<img src="assets/FreiHAND.PNG" width="360" height="264">
</p>

## Troubleshoots
* `RuntimeError: Subtraction, the '-' operator, with a bool tensor is not supported. If you are trying to invert a mask, use the '~' or 'logical_not()' operator instead.`: Go to [here](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527)

## Reference  
```  
@InProceedings{Moon_2022_CVPRW_Hand4Whole,  
author = {Moon, Gyeongsik and Choi, Hongsuk and Lee, Kyoung Mu},  
title = {Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation},  
booktitle = {Computer Vision and Pattern Recognition Workshop (CVPRW)},  
year = {2022}  
}  
```
