import random
import numpy as np
import shutil
import glob
import os
import cv2
import sys
import natsort
import json
from glob import glob
from facenet_pytorch import MTCNN
from PIL import Image
from main.config import cfg
from collections import OrderedDict
from models.Hand_detector import Handtrack
from main.model import bbox_portion_mod
from torch.utils.data.dataset import Dataset
from main.config import cfg
from common.utils.preprocessing import read_json


class MultipleDatasets(Dataset):
    def __init__(self, dbs, make_same_len=True):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num 
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]

def video2sequence2(video_path,total_frame_folder):
    print('extract frames from video: {}...'.format(video_path))
    # videofolder = video_path.split('.')[0]
    # os.makedirs(videofolder, exist_ok=True)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        success,image = vidcap.read()
        if image is None:
            break
        if count % 1 == 0:
            imagepath = '{}/{}_frame{:05d}.jpg'.format(total_frame_folder, video_name, count)
            cv2.imwrite(imagepath, image)     # save frame as JPEG file
            imagepath_list.append(imagepath)
        count += 1

    print('video frames are stored in {}'.format(total_frame_folder))
    return imagepath_list, count


def video2sequence(video_path):
    print('extract frames from video: {}...'.format(video_path))
    videofolder = video_path.split('.')[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        success,image = vidcap.read()
        if image is None:
            break
        if count % 1 == 0:
            imagepath = '{}/{}_frame{:05d}.jpg'.format(videofolder, video_name, count)
            cv2.imwrite(imagepath, image)     # save frame as JPEG file
            imagepath_list.append(imagepath)
        count += 1

    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list, count

def multi_video2sequence(video_path,idx, total_frame_folder):
    print("#############################################################################")
    print('extract frames from video: list {}, {}...'.format(idx+1, video_path))
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        success,image = vidcap.read()
        if image is None:
            break
        if count % 1 == 0:
            imagepath = '{}/{}_{}_frame{:05d}.jpg'.format(total_frame_folder, idx, video_name, count)
            cv2.imwrite(imagepath, image)     # save frame as JPEG file
            imagepath_list.append(imagepath)
        count += 1

    print('video frames are stored in {}'.format(total_frame_folder))
    return imagepath_list, count

def zero_padding(img):
    # Image load (cv2.imread는 BGR로 load 하기에 RGB로 변환)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 가로, 세로에 대해 부족한 margin 계산
    height, width = image.shape[0:2]
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

    # 부족한 길이가 절반으로 안 떨어질 경우 +1
    if np.abs(height - width) % 2 != 0:
        margin[0] += 1

    # 가로, 세로 가운데 부족한 쪽에 margin 추가
    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]

    # color 이미지일 경우 color 채널 margin 추가
    if len(image.shape) == 3:
        margin_list.append([0, 0])

    # 이미지에 margin 추가
    output = np.pad(image, margin_list, mode='constant')
    new_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    # new_output = cv2.resize(new_output,(cfg.output_image_size),interpolation = cv2.INTER_AREA)
    return new_output
    # view

    # save
    # new_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('output.jpg', new_output)

def read_json(json_file):
    with open(json_file, 'r') as tmp_json:
        tmp = json.load(tmp_json)
        v_list = tmp['hand_change']
        return v_list

def make_json(new_json_dir, **kwargs):
    option = kwargs
    file_data = OrderedDict()
    for key, values in option.items():
        file_data[key] = values

    with open(new_json_dir, 'w', encoding='utf-8') as make_file:
        json.dump(file_data, make_file, ensure_ascii=False)


def bbox_IoU(box1, box2):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = box1[0], box1[1], box1[2]+box1[0], box1[3]+box1[1]
    boxB = box2[0], box2[1], box2[2]+box2[0], box2[3]+box2[1]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

class TestData(Dataset):
    ## 수정
    # def __init__(self, testpath, iscrop=False, crop_size=224, hd_size = 1024, scale=1.1, body_detector='rcnn', device='cpu'):
    def __init__(self, testpath, rotation, savefolder, crop_size=224, hd_size = 1024, scale=1.1,device='cpu'):

        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        # testpath = ['/mnt/dms/KTW/data/sign/게다가.mp4','/mnt/dms/KTW/data/sign/젖소.mp4','/mnt/dms/KTW/data/sign/흔들다.mp4','/mnt/dms/KTW/data/sign/ㅔ_2.mp4']
        # testpath = '/mnt/dms/KTW/data/sign/가볍다2'
        # testpath = '/mnt/dms/KTW/data/sign/1920'
        # testpath = '/mnt/dms/KTW/data/sign/ㅔ_2'
        # testpath = '/mnt/dms/KTW/data/sign/ㅅ2'
        # testpath = '/mnt/dms/KTW/data/sign/아줌마2'
        # testpath = '/mnt/dms/KTW/data/sign/때.mp4'
        # testpath = '/mnt/dms/KTW/data/sign/젖소.mp4'

        if os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'mkv', 'csv', 'MOV']):
            video_name = testpath.split('/')[-1].split('.')[0]
            total_frame_folder = os.path.join(savefolder,'frame','original','{}_frame'.format(video_name))
            render_image_save_folder = os.path.join(savefolder,'frame','render',video_name)
            self.mesh_json_save_folder = os.path.join(savefolder,'json')
            if os.path.exists(render_image_save_folder):
                shutil.rmtree(render_image_save_folder)
            os.makedirs(render_image_save_folder, exist_ok=True)
            os.makedirs(self.mesh_json_save_folder, exist_ok=True)

            if os.path.exists(total_frame_folder):
                shutil.rmtree(total_frame_folder)
            os.makedirs(total_frame_folder, exist_ok=True)
            frame_list, count = video2sequence2(testpath,total_frame_folder)
            self.imagepath_list = frame_list
            self.frame_cnt = [count]
            self.render_image_save_folder = render_image_save_folder

        elif os.path.isdir(testpath):
            self.imagepath_list = glob(testpath + '/*.jpg') + glob(testpath + '/*.png') + glob(testpath + '/*.jpeg')
            self.frame_cnt = [len(self.imagepath_list)]
            self.mesh_json_save_folder = os.path.join(savefolder,'json')
            os.makedirs(self.mesh_json_save_folder, exist_ok=True)
            self.render_image_save_folder = os.path.join(savefolder,'frame')


        # else:
        #     os.path.isdir(testpath):
        #     videopath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.jpeg') + glob(testpath +'/*.mp4') + glob(testpath +'/*.MOV')
        #     self.video_cnt = [len(videopath_list)]
        #     total_list = []
        #     frame_cnt = []
        #     for idx,video_path in enumerate(videopath_list):
        #         video_name = video_path.split('/')[-1].split('.')[0]
        #         total_frame_folder = os.path.join(savefolder, '{}_total_frame'.format(video_name))
        #         os.makedirs(total_frame_folder, exist_ok=True)
        #         frame_list, count = multi_video2sequence(video_path, idx, total_frame_folder)
        #         total_list.append(frame_list)
        #         frame_cnt.append(count)
        #     self.videopath_list = total_list
        #     self.frame_cnt = frame_cnt
        else:
            print(f'please check the input path: {testpath}')
            exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = natsort.natsorted(self.imagepath_list)
        self.crop_size = crop_size
        self.hd_size = hd_size
        self.scale = scale
        self.rotation = rotation

        self.bbox_detector = Handtrack(cfg.hand_model_path2,device)
        self.face_model = MTCNN(select_largest=False, device='cuda',  thresholds = [0.6,0.7,0.7])
        self.cnt = 0

        self.before_face_bbox = [0,0,0,0]
        self.before_raw_hand_bboxes = None
        self.before_image_name = str(0)
        self.before_right_hand_bbox = [0,0,0,0]
        self.before_left_hand_bbox = [0,0,0,0]
        
        self.hand_change_list = read_json(cfg.hand_change_list)
        if video_name in self.hand_change_list:
            self.hand_change = 1
        else:
            self.hand_change = 0

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]
        img_original_bgr = cv2.imread(imagepath)
        
        first_image_name = imagename.split("_")[:-1][-1]
        
        if self.rotation:
            img_original_bgr = cv2.rotate(img_original_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if img_original_bgr.shape[1] != img_original_bgr.shape[0]:
            img_original_bgr = zero_padding(img_original_bgr.copy())

        ## face
        face_bbox, _ = self.face_model.detect(img_original_bgr)
        
        if face_bbox is None:
            face_bbox_n = None
        else:
            x1, y1, w, h = face_bbox[0][0], face_bbox[0][1], face_bbox[0][2] - face_bbox[0][0], face_bbox[0][3] - \
                           face_bbox[0][1]

            if w * h <= self.before_face_bbox[2] * self.before_face_bbox[3] * 0.9:
                face_bbox_n = None
            else:
                self.before_face_bbox = x1, y1, w, h
                face_bbox_n = x1, y1, w, h

        ## hand
        ## detect_output = [min_x, min_y, width, height]
        hand_bbox_list = self.bbox_detector.detect_hand_bbox(img_original_bgr.copy(),index)
        # hand_bbox_list, raw_hand_bboxes = detect_output
        #
        #
        ## check raw_hand_bbox
        if len(hand_bbox_list) == 0:
            h_bbox = dict()
            h_bbox['left_hand'] = self.before_lhand_bbox
            h_bbox['right_hand'] = self.before_rhand_bbox
            hand_bbox_list = [h_bbox]
        else:
            self.before_raw_hand_bboxes = hand_bbox_list


        if self.before_image_name != first_image_name:
            self.before_image_name = first_image_name
            self.before_lhand_bbox = hand_bbox_list[0]['left_hand']
            self.before_rhand_bbox = hand_bbox_list[0]['right_hand']


        return {'original_img': img_original_bgr,
                'image_name': imagename,
                'right_hand': hand_bbox_list[0]['right_hand'],
                'left_hand': hand_bbox_list[0]['left_hand'],
                'face': face_bbox_n,
                'cnt': self.cnt,
                'frame_cnt': self.frame_cnt,
                'hand_change' : self.hand_change
                }

