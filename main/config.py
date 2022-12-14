import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset
    trainset_3d = ['Human36M'] 
    trainset_2d = ['MSCOCO', 'MPII']
    testset = 'PW3D'

    ## model setting
    resnet_type = 50
    
    ## training config
    lr = 1e-4
    lr_dec_factor = 10
    train_batch_size = 48

    ## testing config
    test_batch_size = 64

    ## others
    num_thread = 40
    gpu_ids = '0'
    num_gpus = 1
    parts = 'body'
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')

    input_hand_img_shape = (256, 256)
    input_face_img_shape = (256, 192)

    output_image_size = (1920,1920)
    input_image_size = None
    first_cam_trans = None
    first_face_shape = None

    # hand_model_path2 = '/mnt/dms/KTW/hand4whole/weights/model_pretrain.pth'
    # hand_model_path = '/mnt/dms/KTW/hand4whole/weights/snapshot_12_hand.pth.tar'
    # face_model_path = '/mnt/dms/KTW/hand4whole/weights/snapshot_12_face.pth.tar'
    # hand_sign_vocab_path = "/mnt/dms/KTW/hand_sqs/weights/suwha_vocab"
    # hand_sign_ckpt_path = "/mnt/dms/KTW/hand_sqs/weights/output_1800epochs.ckpt"
    # handsign_folder_path = "/mnt/dms/HAND_SIGN/val_demo/"

    hand_track_model = '/mnt/dms/KTW/hand4whole/weights/model_pretrain.pth'
    hand_model_path = '/mnt/dms/KTW/hand4whole/weights/snapshot_12_hand.pth.tar'
    face_model_path = '/mnt/dms/KTW/hand4whole/weights/snapshot_12_face.pth.tar'

    hand_sign_model_conf = "/mnt/dms/KTW/hand_sqs/weights2/seq2seq_BertEncoder.yaml"
    hand_sign_tokenizer_conf = "/mnt/dms/KTW/hand_sqs/weights2/tokenizer.yaml"
    hand_sign_weight_path = "/mnt/dms/KTW/hand_sqs/weights2/best_83epochs.pth"

    handsign_folder_path = "/mnt/dms/HAND_SIGN/val_npz/json"
    hand_change_list = "/mnt/dms/KTW/hand4whole/weights/hand_change_list.json"

    hand_config= osp.join(root_dir,'projects','HandLer','configs','yt_trk.yaml')

    font_dir = "/mnt/dms/KTW/data/font/?????? ??????/????????????/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf"
    frame_cnt_path= "/mnt/dms/HAND_SIGN/saved_val_video/frame/original"
    error_log_save = "/mnt/dms/KTW/hand4whole/error_log"

    def set_args(self, gpu_ids, parts, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.parts = parts
        self.continue_train = continue_train
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

        if self.parts == 'body':
            self.bbox_3d_size = 2
            self.camera_3d_size = 2.5
            self.input_img_shape = (256, 192)
            self.output_hm_shape = (8, 8, 6)
            self.lr_dec_epoch = [4, 6]
            self.end_epoch = 7
        elif self.parts == 'hand':
            self.bbox_3d_size = 0.3  #0.168
            self.camera_3d_size = 0.4
            self.input_img_shape = (256, 256)
            self.output_hm_shape = (8, 8, 8)
            self.lr_dec_epoch = [10, 12] 
            self.end_epoch = 13 
        elif self.parts == 'face':
            self.bbox_3d_size = 0.3  #0.168
            self.camera_3d_size = 0.4
            self.input_img_shape = (256, 192)
            self.output_hm_shape = (8, 8, 6)
            self.lr_dec_epoch = [10, 12] 
            self.end_epoch = 13 
        else:
            assert 0, 'Unknown parts: ' + self.parts
        
        self.focal = (5000, 5000) # virtual focal lengths
        self.princpt = (self.input_img_shape[1]/2, self.input_img_shape[0]/2) # virtual principal point position
        # self.princpt = (self.input_img_shape[1]/2, self.input_img_shape[0]/2) # virtual principal point position

        # self.princpt = (1080/2, 1080/2) # virtual principal point position

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
# from utils.dir import add_pypath, make_folder
# add_pypath(osp.join(cfg.data_dir))
# for i in range(len(cfg.trainset_3d)):
#     add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
# for i in range(len(cfg.trainset_2d)):
#     add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
# add_pypath(osp.join(cfg.data_dir, cfg.testset))
# make_folder(cfg.model_dir)
# make_folder(cfg.vis_dir)
# make_folder(cfg.log_dir)
# make_folder(cfg.result_dir)
