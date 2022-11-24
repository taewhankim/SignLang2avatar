import copy
import sys
import os
import os.path as osp
import argparse
import cv2
import gc
import shutil
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from datetime import datetime, timezone, timedelta
[sys.path.append(os.path.join(os.path.abspath(os.path.dirname('.')),i)) for i in ['.','main','data','common','models','seq2seq_translator']]
# [sys.path.append(i) for i in ['.', '..']]

from main.config import cfg
from main.model import get_model, source_target_portion , make_princpt,make_focal, cont_dict, bbox_portion_mod
from common.utils.preprocessing import generate_patch_image, bbox_preprocessing, process_bbox
from common.utils.human_models import mano, flame
from common.utils.vis import render_mesh
from common.utils.renderer import *
from infer_api import Speech2HandsignFP
from tqdm import tqdm
from data.dataset import TestData
from models.synergynet.uv_texture_realFaces import sy_net

import time
import natsort
from PIL import Image
from glob import glob
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def syn_imgaes(img1,img2,roi_bbox):
    src1 = img1  # 사과파일 읽기
    src2 = img2  # 로고파일 읽기

    rows, cols, channels = src2.shape  # 로고파일 픽셀값 저장
    # roi = src1[50:rows + 50, 50:cols + 50]  # 로고파일 필셀값을 관심영역(ROI)으로 저장함.

    roi = img1[roi_bbox[1]:roi_bbox[1] + roi_bbox[3],\
                            roi_bbox[0]:roi_bbox[0] + roi_bbox[2], :]


    gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)  # 로고파일의 색상을 그레이로 변경
    ret, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # 배경은 흰색으로, 그림을 검정색으로 변경
    mask_inv = cv2.bitwise_not(mask)
    # cv2.imshow('mask', mask)  # 배경 흰색, 로고 검정
    # cv2.imshow('mask_inv', mask_inv)  # 배경 검정, 로고 흰색

    src1_bg = cv2.bitwise_and(roi, roi, mask=mask)  # 배경에서만 연산 = src1 배경 복사
    # cv2.imshow('src1_bg', src1_bg)

    src2_fg = cv2.bitwise_and(src2, src2, mask=mask_inv)  # 로고에서만 연산
    # cv2.imshow('src2_fg', src2_fg)

    dst = cv2.bitwise_or(src1_bg, src2_fg)  # src1_bg와 src2_fg를 합성
    # cv2.imshow('dst', dst)

    src1[roi_bbox[1]:roi_bbox[1] + roi_bbox[3],\
                            roi_bbox[0]:roi_bbox[0] + roi_bbox[2]] = dst

    return src1


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

# cfg.set_args(args.gpu_ids, 'hand')
cudnn.benchmark = True




def load_models(path,parts):
    cfg.set_args(args.gpu_ids, parts)
    model_path = path
    assert osp.exists(model_path), 'Cannot find models at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_model('test', parts)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'], strict=False)
    return model.eval()

hand_model = load_models(cfg.hand_model_path,'hand')
face_model = load_models(cfg.face_model_path,'face')
cfg.set_args(args.gpu_ids, 'hand')


input_list = ['전통 돌하르방도 있고 고인돌도 있어 전통 돌하르방도 있고 고인돌도 있어 제주의 옛 문화를 직접 눈으로 보며 체험할 수 있습니다']
if args.input_text:
    hand_text = []
    hand_videos = []
    # for idx, input_txt in enumerate(args.input_text):
    for idx, input_txt in enumerate(input_list):

        hand_sign_s2s = Speech2HandsignFP(model_conf=cfg.hand_sign_model_conf, weight=cfg.hand_sign_weight_path,
                                          tokenizer_conf=cfg.hand_sign_tokenizer_conf, handsign_folder_path=cfg.handsign_folder_path)
        # hand_sign_output = hand_sign_s2s.translate(args.input_text)
        hand_sign_output = hand_sign_s2s.translate(input_txt)
        hand_text.append(hand_sign_output[0])
        hand_videos += hand_sign_output[1]

    print("########## ORIGINAL COMMENTS    :   ", args.input_text, "     ##########")
    print("########## HAND SIGN TRANSLATIONS    :   ", hand_sign_output[0], "     ##########")


# testdata = TestData(hand_sign_output[1], args.rotation, args.savefolder,device='cuda')
testdata = TestData(hand_videos, args.rotation, args.savefolder,device='cuda')

# save_obj_dir = os.path.join(args.savefolder, 'obj')
# os.makedirs(save_obj_dir, exist_ok=True)

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

before_frame_rhand = None
before_frame_lhand = None

bbox_data = []
tmp = []
tmp_result = []
zero_idx = []
t2 =time.time()
print("preprocess time :",t2-t1)


imgsave = "/mnt/dms/KTW/data/result_img"
if os.path.exists(imgsave):
    shutil.rmtree(imgsave)
os.makedirs(imgsave, exist_ok=True)
cnt=0

for i, batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
    try:
        original_img = batch['original_img']
        video_name = batch['image_name'].split("_")[-2]
        frame_name = batch['image_name'].split("_")[-1]
        hand_change = batch['hand_change']

        rhand_bbox, lhand_bbox = batch['right_hand'], batch['left_hand']
        # # prepare input image
        transform = transforms.ToTensor()
        original_img_height, original_img_width = original_img.shape[:2]

        portion_h = cfg.output_image_size[0] / original_img.shape[0]
        portion_w = cfg.output_image_size[1] / original_img.shape[1]
        portion = [portion_w, portion_h]

        ## 추가
        diff = batch['size_diff']**(1/2)
        size_portion = [diff,diff]
        ##

        targets = {}
        meta_info = {}

        if batch['face'] is None:
            crop_head = crop_head
        else:
            face_bbox_c = batch['face']
            face_bbox_c = process_bbox(face_bbox_c, original_img_width, original_img_height)
            face_bbox_c = list(map(int,face_bbox_c))
        ### 추가
            crop_head = original_img[face_bbox_c[1]:face_bbox_c[1] + face_bbox_c[3], face_bbox_c[0]:face_bbox_c[0] + face_bbox_c[2], :]
            crop_head = cv2.resize(crop_head, (256, 256), interpolation=cv2.INTER_AREA)

        crop_head = cv2.cvtColor(crop_head, cv2.COLOR_BGR2RGB)
        ch_path = os.path.join(args.savefolder, 'crop_head')
        os.makedirs(ch_path, exist_ok=True)
        img_2 = Image.fromarray(crop_head)
        img_2.save(os.path.join(ch_path, "head_{}.png".format(i)), 'PNG')
        # cv2.imwrite(os.path.join(ch_path, "head_{}.png".format(i)), crop_head)
        ###

        ## face prepro
        if batch['face'] is None:
            n_face_bbox = n_face_bbox
            face_mesh = face_mesh
            face_add_cam = face_add_cam

        else:
            face_bbox = batch['face']
            # face_bbox = process_bbox(face_bbox, original_img_width, original_img_height)
            face_bbox = list(map(int,face_bbox))

            face_img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, face_bbox, 1.0, 0.0, False, cfg.input_face_img_shape)
            face_img = transform(face_img.astype(np.float32)) / 255
            face_img = face_img.cuda()[None, :, :, :]
            face_inputs = {'img': face_img}

            with torch.no_grad():
                face_out = face_model(face_inputs, targets, meta_info,i, 'test')
            ## face output
            face_add_cam = face_out['add_cam_trans']
            face_mesh = face_out['flame_mesh_cam'].detach().cpu().numpy()[0]
            face_bbox = bbox_portion_mod(face_bbox, portion)

            ## 추가
            face_bbox = bbox_portion_mod(face_bbox, size_portion)
            ##

            n_face_bbox = copy.deepcopy(np.array(face_bbox))

        if rhand_bbox is None:
            rhand_bbox = ori_rh
            rhand_img = before_frame_rhand_img
        else:
            rhand_img, rhand_bbox = bbox_preprocessing(original_img, rhand_bbox, flip=False)
            before_frame_rhand_img = copy.deepcopy(rhand_img)

        if lhand_bbox is None:
            lhand_bbox = ori_lh
            lhand_img = before_frame_lhand_img
        else:
            lhand_img, lhand_bbox = bbox_preprocessing(original_img, lhand_bbox, flip=True)
            before_frame_lhand_img = copy.deepcopy(lhand_img)

        # hand forward
        img = torch.cat((rhand_img, lhand_img))
        inputs = {'img': img}
        with torch.no_grad():
            out = hand_model(inputs, targets, meta_info,i, 'test')

        ## hand output
        hand_add_cam = out['add_cam_trans']

        rhand_mesh = out['mano_mesh_cam'].detach().cpu().numpy()[0]
        lhand_mesh = out['mano_mesh_cam'].detach().cpu().numpy()[1]
        lhand_mesh[:, 0] *= -1  # flip back to the left hand mesh

        ## 추가
        ori_rh = copy.deepcopy(rhand_bbox)
        ori_lh = copy.deepcopy(lhand_bbox)
        ##

        rhand_bbox = bbox_portion_mod(rhand_bbox,portion)
        lhand_bbox = bbox_portion_mod(lhand_bbox,portion)

        ## 추가
        rhand_bbox = bbox_portion_mod(rhand_bbox,size_portion)
        lhand_bbox = bbox_portion_mod(lhand_bbox,size_portion)
        ##

        n_rhand_bbox = copy.deepcopy(np.array(rhand_bbox))
        n_lhand_bbox = copy.deepcopy(np.array(lhand_bbox))


        result_data = {"face_mesh": face_mesh,
                       "face_add_cam":face_add_cam,
                       "face_bbox": n_face_bbox,
                       "rhand_mesh": rhand_mesh,
                       "lhand_mesh": lhand_mesh,
                       "hand_add_cam": hand_add_cam,
                       "rhand_bbox":n_rhand_bbox,
                       "lhand_bbox":n_lhand_bbox,
                       "hand_change": hand_change
                       }

        tmp_result,check = cont_dict(args,result_data,tmp,tmp_result)

        # del rhand_bbox
        # del lhand_bbox
        del out
        del rhand_mesh
        del lhand_mesh
        del result_data
        gc.collect()

    except Exception as e:
        print("error message :   ", e)
        error_folder = os.path.join(cfg.error_log_save, _day)
        os.makedirs(error_folder, exist_ok=True)
        error_path = os.path.join(error_folder, "error_video_frame.txt")
        if not os.path.isfile(error_path):
            file_txt = open(error_path, "w", encoding="UTF-8")
            file_txt.close()
        file_txt = open(error_path, "a", encoding="UTF-8")
        file_txt.write("video_list : {},  frame_name : {},  time : {},  ---  error message : {}\n".format(video_name, frame_name, datetime.now(KST), e))
        file_txt.close()
        continue

    # if os.path.isfile(args.inputpath) and (args.inputpath[-3:] in ['mp4', 'csv', 'MOV']):
    #     vidcap= cv2.VideoCapture(args.inputpath)
    #     fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    #     # frame_width = round(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     # frame_height = round(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     frame_width = 1080
    #     frame_height = 650
    # else:

    ### make video
fps = 60
frame_width = cfg.output_image_size[1]
frame_height = int(cfg.output_image_size[1]*0.56)+150 #1080


#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#outcap_only_hand = cv2.VideoWriter('{}/{}.mp4'.format(args.savefolder, "only_hand"),
#                               fourcc, fps, (frame_height, frame_width))

### make white image
vis_img = np.ones([cfg.output_image_size[0],cfg.output_image_size[1], 3]) * 255


### make substitle
# substitle = np.ones([150,cfg.output_image_size[1],3],np.uint8)*255
# substitle2 = Image.fromarray(substitle)
# draw = ImageDraw.Draw(substitle2)
# fonts = ImageFont.truetype(cfg.font_dir,50)
# f_w, f_h = fonts.getsize(args.input_text)
# draw.text((int((substitle.shape[1]-f_w)/2),(int(substitle.shape[0]-f_h)/2) ), args.input_text, font=fonts, fill=(0,0,0))
# substitle = np.array(substitle2)

##
t_c_img = glob(ch_path+"/*.png")
t_c_img = natsort.natsorted(t_c_img)
total_frame_length = testdata.__len__()
contract = len(tmp_result)
trim_size = total_frame_length - contract
for_trim = trim_size // 2
backward_trim = trim_size - for_trim
total_batch = t_c_img[for_trim:-backward_trim]


## fix princpt
fix_princpt = [int(cfg.output_image_size[1]*0.5), int(cfg.output_image_size[0]*0.19)]

## face detect model

from facenet_pytorch import MTCNN
tmp_face_model = MTCNN(select_largest=False, device='cuda', thresholds=[0.6, 0.7, 0.7])
synergynet = sy_net()
#####
t3 = time.time()
print("model time : ",t3-t2)
print("##########################    mesh rendering    ##########################")
for i , (bat,tot) in enumerate(tqdm(zip(tmp_result,total_batch), dynamic_ncols=True)):
    # try:
        # vis_img = original_img.copy()
        # vis_img = np.ones([vis_img.shape[0],vis_img.shape[1],vis_img.shape[2]])*255
        face_mesh = bat['face_mesh']
        face_mesh = bat['face_mesh']
        face_add_cam = bat['face_add_cam']
        rhand_mesh = bat['rhand_mesh']
        lhand_mesh = bat['lhand_mesh']
        hand_add_cam = bat['hand_add_cam']
        hand_change = bat['hand_change']
        face_bbox = bat['face_bbox'].tolist()
        rhand_bbox = bat['rhand_bbox'].tolist()
        lhand_bbox = bat['lhand_bbox'].tolist()

        init_face_princpt = (cfg.input_face_img_shape[1]/2, cfg.input_face_img_shape[0]/2)
        init_hand_princpt = (cfg.input_hand_img_shape[1]/2, cfg.input_hand_img_shape[0]/2)

    ###################
        ori_focal_face = make_focal(cfg.focal,cfg.input_face_img_shape,face_bbox)
        ori_focal_rh = make_focal(cfg.focal,cfg.input_hand_img_shape,rhand_bbox)
        ori_focal_lh = make_focal(cfg.focal,cfg.input_hand_img_shape,lhand_bbox)


        ori_princpt_face = make_princpt(init_face_princpt,cfg.input_face_img_shape,face_bbox)
        ori_princpt_rh = make_princpt(init_hand_princpt,cfg.input_hand_img_shape,rhand_bbox)
        ori_princpt_lh = make_princpt(init_hand_princpt,cfg.input_hand_img_shape,lhand_bbox)

        ori_face_fix_face = source_target_portion(ori_princpt_face, fix_princpt)

        por_princpt_face = fix_princpt
        por_princpt_rh = [ori_face_fix_face[0]+ori_princpt_rh[0],ori_face_fix_face[1]+ori_princpt_rh[1]]
        por_princpt_lh = [ori_face_fix_face[0]+ori_princpt_lh[0],ori_face_fix_face[1]+ori_princpt_lh[1]]

        # face_rendered_img = render_mesh(vis_img, face_mesh, flame.face, {'focal': ori_focal_face, 'princpt': por_princpt_face, 'add_cam_trans': face_add_cam})
        face_rendered_img = vis_img
        crop_head = cv2.imread(tot)

        face_rendered_img2 = synergynet(crop_head,face_bbox)
        new_f_h,new_f_w = face_rendered_img2.shape[:2]
        new_face_bbox = fix_princpt[0]- new_f_w/2, fix_princpt[1] - new_f_h/2, new_f_w, new_f_h
        new_face_bbox = list(map(int, new_face_bbox))
        if i ==0:
            first_bbox = new_face_bbox
            first_size = face_rendered_img2.shape[:2]
        else:
            new_face_bbox = first_bbox
            face_rendered_img2 = cv2.resize(face_rendered_img2,(int(first_size[1]),int(first_size[0])),interpolation=cv2.INTER_AREA)

        face_rendered_img = syn_imgaes(face_rendered_img.astype(np.uint8),face_rendered_img2,new_face_bbox)
        # face_rendered_img[new_face_bbox[1]:new_face_bbox[1]+new_face_bbox[3],new_face_bbox[0]:new_face_bbox[0]+new_face_bbox[2],:] =face_rendered_img2

        if hand_change != 0:
            right_rendered_img = render_mesh(face_rendered_img, rhand_mesh, mano.face['right'],
                                             {'focal': ori_focal_rh, 'princpt': por_princpt_rh,
                                              'add_cam_trans': hand_add_cam[0].unsqueeze(0)})
            final_rendered_img = render_mesh(right_rendered_img, lhand_mesh, mano.face['left'],
                                            {'focal': ori_focal_lh, 'princpt': por_princpt_lh,
                                             'add_cam_trans': hand_add_cam[1].unsqueeze(0)})
        else:
            left_rendered_img = render_mesh(face_rendered_img, lhand_mesh, mano.face['left'], {'focal': ori_focal_lh , 'princpt': por_princpt_lh, 'add_cam_trans': hand_add_cam[1].unsqueeze(0)})
            final_rendered_img = render_mesh(left_rendered_img, rhand_mesh, mano.face['right'], {'focal': ori_focal_rh , 'princpt': por_princpt_rh,'add_cam_trans': hand_add_cam[0].unsqueeze(0)})

        # final_rendered_img = final_rendered_img.astype(np.uint8).copy()
        final_rendered_img = final_rendered_img.astype(np.uint8).copy()[:int(cfg.output_image_size[0]*0.7),:,:]
        # cv2.imwrite(os.path.join(save_dir, 'render_original_img_{0:05d}.jpg').format(i), final_rendered_img)

        # new_final_rendered_img = np.concatenate((final_rendered_img,substitle), axis=0)
        cv2.imwrite(os.path.join(save_dir, 'render_original_img_{0:05d}.jpg').format(i), final_rendered_img)

    # except Exception as e:
    #     print("error message :   ", e)
    #     error_folder = os.path.join(cfg.error_log_save, _day)
    #     os.makedirs(error_folder, exist_ok=True)
    #     error_path = os.path.join(error_folder, "error_mesh.txt")
    #     if not os.path.isfile(error_path):
    #         file_txt = open(error_path, "w", encoding="UTF-8")
    #         file_txt.close()
    #     file_txt = open(error_path, "a", encoding="UTF-8")
    #     file_txt.write(
    #         "mesh_idx : {}, time : {}, --- error message : {}\n".format(i, datetime.now(KST), e))
    #     file_txt.close()
    #     continue
t4=time.time()

print("rendering time : ",t4-t3)
print("Total time : ",t4-t1)
ffmpeg_cmd2 = f'ffmpeg -y -f image2 -framerate 60 -pattern_type glob -i "{in_dir2}/*.jpg"  -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {out_path2}'
os.system(ffmpeg_cmd2)




# save MANO parameters
#
#
# mano_pose = out['mano_pose'].detach().cpu().numpy();
# mano_shape = out['mano_shape'].detach().cpu().numpy();
# rmano_pose, rmano_shape = mano_pose[0], mano_shape[0]
# with open(os.path.join(save_json,'mano_param_rhand.json'), 'w') as f:
#     json.dump({'pose': rmano_pose.reshape(-1).tolist(), 'shape': rmano_shape.reshape(-1).tolist()}, f)
# lmano_pose, lmano_shape = mano_pose[1], mano_shape[1]
# lmano_pose = lmano_pose.reshape(-1, 3)
# lmano_pose[:, 1:3] *= -1
# with open(os.path.join(save_json,'mano_param_lhand.json'), 'w') as f:
#     json.dump({'pose': lmano_pose.reshape(-1).tolist(), 'shape': lmano_shape.reshape(-1).tolist()}, f)