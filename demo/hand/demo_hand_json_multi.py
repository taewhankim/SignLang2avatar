import copy
import sys
import os
import os.path as osp
import argparse
import cv2
import gc
import re
import shutil
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from datetime import datetime, timezone, timedelta
[sys.path.append(os.path.join(os.path.abspath(os.path.dirname('.')),i)) for i in ['.','main','data','common','models','seq2seq_translator']]
# [sys.path.append(i) for i in ['.', '..']]

from main.config import cfg
from main.model import bbox_portion_mod, wh_portion_mod
from common.utils.preprocessing import generate_patch_image, bbox_preprocessing, process_bbox
from common.utils.human_models import mano, flame
from common.utils.vis import render_mesh
from common.utils.renderer import *
from infer_api import Speech2HandsignFP
from tqdm import tqdm
from common.utils.vis import rendering_3d
from data.dataset import TestData
import time
from collections import Counter
import multiprocessing as mp
from multiprocessing import Pool
import logging

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    # parser.add_argument('-p', '--inputpath', default='', type=str,
    #                     help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('--input_text', default=None,type=str, help = "input hand_sign sentences")
    parser.add_argument('-s', '--savefolder', default='', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='set device, cpu for using cpu')
    parser.add_argument('--rotation', default=False, action='store_true', help='rotation 270degree')
    parser.add_argument('--cont_n', default=None, type=int,
                        help='window_size' )
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

t1 = time.time()

args = parse_args()
os.makedirs(args.savefolder, exist_ok=True)

cudnn.benchmark = True
cfg.set_args(args.gpu_ids, 'hand')

def read_npy(npy_file):
    ann = np.load(npy_file,allow_pickle=True)
    frame_cnt = ann['video'].tolist()['frame_cnt']
    video_name= ann['video'].tolist()['video_name'].split(".")[0]
    output_image_size = ann['mesh'].tolist()['output_image_size']
    mesh_info = ann['mesh'].tolist()['mesh_info']
    return frame_cnt, video_name, output_image_size, mesh_info

def multi_render(ext2):
    # c_proc = mp.current_process()
    # print("@@@@@@ Running on Process",c_proc.name,"PID",c_proc.pid)
    idx, ext = ext2
    c = Counter()
    for d in ext:
        c.update(d)
    single_frame = {kk: vv / args.cont_n for kk, vv in c.items()}
    if idx> first_cnt:
        diff = (first_face_bbox[2]*first_face_bbox[3]/(single_frame['face_bbox'][2]*single_frame['face_bbox'][3]))**(1/2)
        size_portion = [diff,diff]
        single_frame['face_bbox'] = np.array(bbox_portion_mod(single_frame['face_bbox'],size_portion))
        single_frame['rhand_bbox'] = np.array(bbox_portion_mod(single_frame['rhand_bbox'],size_portion))
        single_frame['lhand_bbox'] = np.array(bbox_portion_mod(single_frame['lhand_bbox'],size_portion))
    result_img = rendering_3d(idx, single_frame, vis_img, fix_princpt)
    cv2.imwrite(os.path.join(save_dir, 'render_original_img_{0:05d}.jpg').format(idx), result_img)
    # print("###### Ended index",idx,"Process",c_proc.name)



# input_list = ['전통 돌하르방도 있고 고인돌도 있어 제주의 옛 문화를 직접 눈으로 보며 체험할 수 있습니다']
if args.input_text:
    hand_text = []
    hand_videos = []
    new_list = re.sub(r"[\"\'\[\]]", "", args.input_text)
    new_list = new_list.split(",")
    # for idx, input_txt in enumerate(args.input_text):
    for idx, input_txt in enumerate(new_list):

        hand_sign_s2s = Speech2HandsignFP(model_conf=cfg.hand_sign_model_conf, weight=cfg.hand_sign_weight_path,
                                          tokenizer_conf=cfg.hand_sign_tokenizer_conf, handsign_folder_path=cfg.handsign_folder_path)
        # hand_sign_output = hand_sign_s2s.translate(args.input_text)
        hand_sign_output = hand_sign_s2s.translate(input_txt)
        hand_text.append(hand_sign_output[0])
        hand_videos += hand_sign_output[1]

    print("########## ORIGINAL COMMENTS    :   ", new_list, "     ##########")
    print("########## HAND SIGN TRANSLATIONS    :   ", hand_text, "     ##########")
    for num,path in enumerate(hand_videos):
        print("########## No.{} Path     :    ".format(num),path, "       ###########")



save_dir = os.path.join(args.savefolder, 'video_frame')
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

## 시간
KST = timezone(timedelta(hours=9))
time_record = datetime.now(KST)
_day = str(time_record)[:10]

in_dir2 = save_dir
out_path2 = os.path.join(args.savefolder, "result.mp4")



imgsave = "/mnt/dms/KTW/data/result_img"
if os.path.exists(imgsave):
    shutil.rmtree(imgsave)
os.makedirs(imgsave, exist_ok=True)
# hand_videos = ['/mnt/dms/HAND_SIGN/val_npz/json/귀엽다.npz','/mnt/dms/HAND_SIGN/val_npz/json/조상.npz' ]
# hand_videos = ['/mnt/dms/HAND_SIGN/val_npz/json/전통.npz']

vis_img = np.ones([cfg.output_image_size[0],cfg.output_image_size[1], 3]) * 255
fix_princpt = [int(cfg.output_image_size[1]*0.5), int(cfg.output_image_size[0]*0.19)]

ext = []
cnt = 0
start_n = 0
# enumerate(tqdm(testdata, dynamic_ncols=True))
print("##########################    mesh rendering    ##########################")

# def multi_run_wrapper(args):
#    return multi_render(*args)
# hand_videos = ['/mnt/dms/HAND_SIGN/val_npz/json/ㄱ.npz']
if __name__ == '__main__':
    t3 = time.time()
    proc = mp.current_process()
    print("########     Main P name :  ", proc.name, "   Main P PID :   ", proc.pid)
    for idx, vi in tqdm(enumerate(hand_videos), dynamic_ncols=True):
        try:
            start_time = time.time()
            num_cores = mp.cpu_count()
            frame_cnt, vid_name, out_img_size, mesh_info = read_npy(vi)
            print("\n")
            print("##########################################################################")
            print(f'######  NO. {idx}/{(len(hand_videos)-1)}      Video : {vid_name}, frame : {frame_cnt}   is converting..... ######')
            print("##########################################################################\n")

            if out_img_size != cfg.output_image_size:
                portion_h, portion_w = out_img_size[0] / cfg.output_image_size[0], out_img_size[1] / cfg.output_image_size[
                    1]
                for m_i in mesh_info:
                    m_i['face_bbox'] = bbox_portion_mod(m_i['face_bbox'], (portion_h, portion_w))
                    m_i['rhand_bbox'] = bbox_portion_mod(m_i['rhand_bbox'], (portion_h, portion_w))
                    m_i['lhand_bbox'] = bbox_portion_mod(m_i['lhand_bbox'], (portion_h, portion_w))

            ### fix with first video bbox size
            if idx ==0:
                first_cnt = frame_cnt - args.cont_n + 1
                first_face_bbox = mesh_info[0]['face_bbox']

            ext.extend(mesh_info)
            temp = [ext[i:i+args.cont_n] for i in range(len(ext)-args.cont_n+1)]

            ## log

            # mp.log_to_stderr()
            # logger = mp.get_logger()
            # logger.setLevel(logging.DEBUG)
            # with Pool(num_cores) as pool:
            #     with tqdm(total=len(temp)) as pbar:
            #         for _ in tqdm(pool.imap_unordered(multi_render, list(enumerate(temp,start= start_n)))):
            #             pbar.update()
            pool = Pool(num_cores)
            pool.map(multi_render, list(enumerate(temp,start= start_n)))
            pool.close()
            pool.join()
            start_n = start_n + len(mesh_info) - args.cont_n + 1
            rest_info = mesh_info[frame_cnt - args.cont_n + 1:]
            ext = []
            ext.extend(rest_info)
            del mesh_info
            pool.terminate()
            end_time = time.time()
            print(f'-------  time :  {round(end_time - start_time,3)}')


        except Exception as e:
            print("error message :   ", e)
            error_folder = os.path.join(cfg.error_log_save, _day)
            os.makedirs(error_folder, exist_ok=True)
            error_path = os.path.join(error_folder, "error_mesh.txt")
            if not os.path.isfile(error_path):
                file_txt = open(error_path, "w", encoding="UTF-8")
                file_txt.close()
            file_txt = open(error_path, "a", encoding="UTF-8")
            file_txt.write(
                "mesh_idx : {}, video_name :  {}  time : {}, --- error message : {}\n".format(idx,vid_name, datetime.now(KST), e))
            file_txt.close()
            continue

    t4 = time.time()

    print("total time :    ", t4-t3)
    print("Total time : ",t4-t1)
    ffmpeg_cmd2 = f'ffmpeg -y -f image2 -framerate 60 -pattern_type glob -i "{in_dir2}/*.jpg"  -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {out_path2}'
    os.system(ffmpeg_cmd2)

