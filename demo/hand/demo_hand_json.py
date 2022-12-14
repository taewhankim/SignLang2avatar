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
import time
from collections import Counter

os.environ['PYOPENGL_PLATFORM'] = 'egl'
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

# cfg.set_args(args.gpu_ids, 'hand')
cudnn.benchmark = True
# def load_models(path,parts):
#     cfg.set_args(args.gpu_ids, parts)
#     model_path = path
#     assert osp.exists(model_path), 'Cannot find models at ' + model_path
#     print('Load checkpoint from {}'.format(model_path))
#     model = get_model('test', parts)
#     model = DataParallel(model).cuda()
#     ckpt = torch.load(model_path)
#     model.load_state_dict(ckpt['network'], strict=False)
#     return model.eval()

# hand_model = load_models(cfg.hand_model_path,'hand')
# face_model = load_models(cfg.face_model_path,'face')
cfg.set_args(args.gpu_ids, 'hand')

def read_npy(npy_file):
    ann = np.load(npy_file,allow_pickle=True)
    frame_cnt = ann['video'].tolist()['frame_cnt']
    video_name= ann['video'].tolist()['video_name'].split(".")[0]
    output_image_size = ann['mesh'].tolist()['output_image_size']
    mesh_info = ann['mesh'].tolist()['mesh_info']
    return frame_cnt, video_name, output_image_size, mesh_info

def make_mesh(bin_list,frame_cnt, mesh_info, cont_n):
    # final_mesh_info = []
    for i in range(frame_cnt):
        if i <= frame_cnt-cont_n:
            m_i = mesh_info[i:i+cont_n]
            c = Counter()
            for d in m_i:
                c.update(d)
            bin_list.append({kk:vv/cont_n for kk,vv in c.items()})
        else:
            break
    return bin_list, mesh_info[frame_cnt-cont_n+1:]

# input_list = ['?????? ???????????? ?????? ??????????????? ????????? ???????????? ??????????????????', '????????? ????????? ??????????????? ??? ???????????? ????????????', '???????????? ????????? ????????? ?????? ???????????? ???????????? ????????????', '??? ????????? ????????? ?????? ????????? ????????? ??? ????????? ??? ???????????? ????????????']
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
    print("########## HAND SIGN TRANSLATIONS    :   ", hand_sign_output[0], "     ##########")
    for num,path in enumerate(hand_videos):
        print("########## No.{} Path     :    ".format(num),path, "       ###########")


# testdata = TestData(hand_sign_output[1], args.rotation, args.savefolder,device='cuda')
# testdata = TestData(hand_videos, args.rotation, args.savefolder,device='cuda')

# save_obj_dir = os.path.join(args.savefolder, 'obj')
# os.makedirs(save_obj_dir, exist_ok=True)

save_dir = os.path.join(args.savefolder, 'video_frame')
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

## ??????
KST = timezone(timedelta(hours=9))
time_record = datetime.now(KST)
_day = str(time_record)[:10]

in_dir2 = save_dir
out_path2 = os.path.join(args.savefolder, "result.mp4")

before_frame_rhand = None
before_frame_lhand = None

t2 =time.time()
print("preprocess time :",t2-t1)


imgsave = "/mnt/dms/KTW/data/result_img"
if os.path.exists(imgsave):
    shutil.rmtree(imgsave)
os.makedirs(imgsave, exist_ok=True)
cnt=0
hand_videos = ['/mnt/dms/KTW/hand4whole/results/json_sil3/json/?????????.npz','/mnt/dms/KTW/hand4whole/results/json_sil3/json/??????.npz' ]
# hand_videos = ['/mnt/dms/KTW/hand4whole/results/json_sil3/json/?????????.npz']
''' ### case 1
# final_mesh_info, ext = [],[]
# for i in hand_videos:
#     frame_cnt, vid_name, out_img_size, mesh_info = read_npy(i)
#     if vid_name != i.split("/")[-1].split(".")[0]:
#         print("wrong npz ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#         break
#     else:
#         ext.extend(mesh_info)
#         mesh_result, rest_info = make_mesh(final_mesh_info,len(ext), ext, args.cont_n)
#         ext=[]
#         ext.extend(rest_info)
'''


def rendering_3d(bat,vis_img,fix_princpt):
    try:
        face_mesh = torch.tensor(bat['face_mesh'])
        face_add_cam = torch.tensor(bat['face_add_cam'])
        rhand_mesh = torch.tensor(bat['rhand_mesh'])
        lhand_mesh = torch.tensor(bat['lhand_mesh'])
        hand_add_cam = torch.tensor(bat['hand_add_cam'])
        hand_change = bat['hand_change']
        face_bbox = bat['face_bbox'].tolist()
        rhand_bbox = bat['rhand_bbox'].tolist()
        lhand_bbox = bat['lhand_bbox'].tolist()

        init_face_princpt = (cfg.input_face_img_shape[1] / 2, cfg.input_face_img_shape[0] / 2)
        init_hand_princpt = (cfg.input_hand_img_shape[1] / 2, cfg.input_hand_img_shape[0] / 2)

        ###################
        ori_focal_face = make_focal(cfg.focal, cfg.input_face_img_shape, face_bbox)
        ori_focal_rh = make_focal(cfg.focal, cfg.input_hand_img_shape, rhand_bbox)
        ori_focal_lh = make_focal(cfg.focal, cfg.input_hand_img_shape, lhand_bbox)

        ori_princpt_face = make_princpt(init_face_princpt, cfg.input_face_img_shape, face_bbox)
        ori_princpt_rh = make_princpt(init_hand_princpt, cfg.input_hand_img_shape, rhand_bbox)
        ori_princpt_lh = make_princpt(init_hand_princpt, cfg.input_hand_img_shape, lhand_bbox)

        ori_face_fix_face = source_target_portion(ori_princpt_face, fix_princpt)

        por_princpt_face = fix_princpt
        por_princpt_rh = [ori_face_fix_face[0] + ori_princpt_rh[0], ori_face_fix_face[1] + ori_princpt_rh[1]]
        por_princpt_lh = [ori_face_fix_face[0] + ori_princpt_lh[0], ori_face_fix_face[1] + ori_princpt_lh[1]]

        face_rendered_img = render_mesh(vis_img, face_mesh, flame.face,
                                        {'focal': ori_focal_face, 'princpt': por_princpt_face,
                                         'add_cam_trans': face_add_cam})
        if hand_change != 0:
            right_rendered_img = render_mesh(face_rendered_img, rhand_mesh, mano.face['right'],
                                             {'focal': ori_focal_rh, 'princpt': por_princpt_rh,
                                              'add_cam_trans': hand_add_cam[0].unsqueeze(0)})
            final_rendered_img = render_mesh(right_rendered_img, lhand_mesh, mano.face['left'],
                                             {'focal': ori_focal_lh, 'princpt': por_princpt_lh,
                                              'add_cam_trans': hand_add_cam[1].unsqueeze(0)})
        else:
            left_rendered_img = render_mesh(face_rendered_img, lhand_mesh, mano.face['left'],
                                            {'focal': ori_focal_lh, 'princpt': por_princpt_lh,
                                             'add_cam_trans': hand_add_cam[1].unsqueeze(0)})
            final_rendered_img = render_mesh(left_rendered_img, rhand_mesh, mano.face['right'],
                                             {'focal': ori_focal_rh, 'princpt': por_princpt_rh,
                                              'add_cam_trans': hand_add_cam[0].unsqueeze(0)})

        # final_rendered_img = final_rendered_img.astype(np.uint8).copy()
        final_rendered_img = final_rendered_img.astype(np.uint8).copy()[:int(cfg.output_image_size[0] * 0.7), :, :]
        return final_rendered_img
        # cv2.imwrite(os.path.join(save_dir, 'render_original_img_{0:05d}.jpg').format(i), final_rendered_img)

        # new_final_rendered_img = np.concatenate((final_rendered_img,substitle), axis=0)
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
            "mesh_idx : {}, time : {}, --- error message : {}\n".format(i, datetime.now(KST), e))
        file_txt.close()



vis_img = np.ones([cfg.output_image_size[0],cfg.output_image_size[1], 3]) * 255
fix_princpt = [int(cfg.output_image_size[1]*0.5), int(cfg.output_image_size[0]*0.19)]

ext = []
cnt = 0
# enumerate(tqdm(testdata, dynamic_ncols=True))
t3 = time.time()
print("model time : ",t3-t2)
print("##########################    mesh rendering    ##########################")

for vi in tqdm(hand_videos, dynamic_ncols=True):
    vi = '/mnt/dms/KTW/data/npz/??????.npz'
    frame_cnt, vid_name, out_img_size, mesh_info = read_npy(vi)
    if out_img_size!=cfg.output_image_size:
        portion_h,portion_w = out_img_size[0]/cfg.output_image_size[0], out_img_size[1]/cfg.output_image_size[1]
        mesh_info['face_bbox'] = bbox_portion_mod(mesh_info['face_bbox'],(portion_h,portion_w))
        mesh_info['rhand_bbox'] = bbox_portion_mod(mesh_info['rhand_bbox'],(portion_h,portion_w))
        mesh_info['lhand_bbox'] = bbox_portion_mod(mesh_info['lhand_bbox'],(portion_h,portion_w))
    ext.extend(mesh_info)
    for i in tqdm(range(len(ext)),dynamic_ncols=True):
        if i <= len(ext)-args.cont_n:
            m_i = ext[i:i+args.cont_n]
            c = Counter()
            for d in m_i:
                c.update(d)
            single_frame = {kk:vv/args.cont_n for kk,vv in c.items()}
            result_img = rendering_3d(single_frame,vis_img,fix_princpt)
            cv2.imwrite(os.path.join(save_dir, 'render_original_img_{0:05d}.jpg').format(cnt), result_img)
            cnt+=1
        else:
            ext = []
            rest_info = mesh_info[frame_cnt-args.cont_n+1:]
            ext.extend(rest_info)
            break

t4 = time.time()

print("total time :    ", t4-t3)




#
# else:
#     frame_cnt , vid_name, out_img_size, mesh_info = read_npy(hand_videos[0])
#     if vid_name != hand_videos[0].split("/")[-1].split(".")[0]:
#         print("wrong npz ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     else:
#         mesh_result, _ = make_mesh(final_mesh_info,frame_cnt,mesh_info, args.cont_n)
#
# for i, batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
#     try:
#         original_img = batch['original_img']
#         video_name = batch['image_name'].split("_")[-2]
#         frame_name = batch['image_name'].split("_")[-1]
#         hand_change = batch['hand_change']
#
#         rhand_bbox, lhand_bbox = batch['right_hand'], batch['left_hand']
#         # # prepare input image
#         transform = transforms.ToTensor()
#         original_img_height, original_img_width = original_img.shape[:2]
#
#         portion_h = cfg.output_image_size[0] / original_img.shape[0]
#         portion_w = cfg.output_image_size[1] / original_img.shape[1]
#         portion = [portion_w, portion_h]
#
#         ## ??????
#         diff = batch['size_diff']**(1/2)
#         size_portion = [diff,diff]
#         ##
#
#         targets = {}
#         meta_info = {}
#         ## face prepro
#
#         if batch['face'] is None:
#             n_face_bbox = n_face_bbox
#             face_mesh = face_mesh
#             face_add_cam = face_add_cam
#
#         else:
#             face_bbox = batch['face']
#             # face_bbox = process_bbox(face_bbox, original_img_width, original_img_height)
#             face_bbox = list(map(int,face_bbox))
#
#             face_img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, face_bbox, 1.0, 0.0, False, cfg.input_face_img_shape)
#             face_img = transform(face_img.astype(np.float32)) / 255
#             face_img = face_img.cuda()[None, :, :, :]
#             face_inputs = {'img': face_img}
#
#             with torch.no_grad():
#                 face_out = face_model(face_inputs, targets, meta_info,i, 'test')
#             ## face output
#             face_add_cam = face_out['add_cam_trans']
#             face_mesh = face_out['flame_mesh_cam'].detach().cpu().numpy()[0]
#             face_bbox = bbox_portion_mod(face_bbox, portion)
#
#             ## ??????
#             face_bbox = bbox_portion_mod(face_bbox, size_portion)
#             ##
#
#             n_face_bbox = copy.deepcopy(np.array(face_bbox))
#
#         if rhand_bbox is None:
#             rhand_bbox = ori_rh
#             rhand_img = before_frame_rhand_img
#         else:
#             rhand_img, rhand_bbox = bbox_preprocessing(original_img, rhand_bbox, flip=False)
#             before_frame_rhand_img = copy.deepcopy(rhand_img)
#
#         if lhand_bbox is None:
#             lhand_bbox = ori_lh
#             lhand_img = before_frame_lhand_img
#         else:
#             lhand_img, lhand_bbox = bbox_preprocessing(original_img, lhand_bbox, flip=True)
#             before_frame_lhand_img = copy.deepcopy(lhand_img)
#
#         # hand forward
#         img = torch.cat((rhand_img, lhand_img))
#         inputs = {'img': img}
#         with torch.no_grad():
#             out = hand_model(inputs, targets, meta_info,i, 'test')
#
#         ## hand output
#         hand_add_cam = out['add_cam_trans']
#
#         rhand_mesh = out['mano_mesh_cam'].detach().cpu().numpy()[0]
#         lhand_mesh = out['mano_mesh_cam'].detach().cpu().numpy()[1]
#         lhand_mesh[:, 0] *= -1  # flip back to the left hand mesh
#
#         ## ??????
#         ori_rh = copy.deepcopy(rhand_bbox)
#         ori_lh = copy.deepcopy(lhand_bbox)
#         ##
#
#         rhand_bbox = bbox_portion_mod(rhand_bbox,portion)
#         lhand_bbox = bbox_portion_mod(lhand_bbox,portion)
#
#         ## ??????
#         rhand_bbox = bbox_portion_mod(rhand_bbox,size_portion)
#         lhand_bbox = bbox_portion_mod(lhand_bbox,size_portion)
#         ##
#
#         n_rhand_bbox = copy.deepcopy(np.array(rhand_bbox))
#         n_lhand_bbox = copy.deepcopy(np.array(lhand_bbox))
#
#
#         result_data = {"face_mesh": face_mesh,
#                        "face_add_cam":face_add_cam,
#                        "face_bbox": n_face_bbox,
#                        "rhand_mesh": rhand_mesh,
#                        "lhand_mesh": lhand_mesh,
#                        "hand_add_cam": hand_add_cam,
#                        "rhand_bbox":n_rhand_bbox,
#                        "lhand_bbox":n_lhand_bbox,
#                        "hand_change": hand_change
#                        }
#         tmp_result,check = cont_dict(args,result_data,tmp,tmp_result)
#
#         # del rhand_bbox
#         # del lhand_bbox
#         del out
#         del rhand_mesh
#         del lhand_mesh
#         del result_data
#         gc.collect()
#
#     except Exception as e:
#         print("error message :   ", e)
#         error_folder = os.path.join(cfg.error_log_save, _day)
#         os.makedirs(error_folder, exist_ok=True)
#         error_path = os.path.join(error_folder, "error_video_frame.txt")
#         if not os.path.isfile(error_path):
#             file_txt = open(error_path, "w", encoding="UTF-8")
#             file_txt.close()
#         file_txt = open(error_path, "a", encoding="UTF-8")
#         file_txt.write("video_list : {},  frame_name : {},  time : {},  ---  error message : {}\n".format(video_name, frame_name, datetime.now(KST), e))
#         file_txt.close()
#         continue

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

## fix princpt
fix_princpt = [int(cfg.output_image_size[1]*0.5), int(cfg.output_image_size[0]*0.19)]

#####



# for i , bat in enumerate(tqdm(tmp_result, dynamic_ncols=True)):
#     try:
#         # vis_img = original_img.copy()
#         # vis_img = np.ones([vis_img.shape[0],vis_img.shape[1],vis_img.shape[2]])*255
#         face_mesh = bat['face_mesh']
#         face_mesh = bat['face_mesh']
#         face_add_cam = bat['face_add_cam']
#         rhand_mesh = bat['rhand_mesh']
#         lhand_mesh = bat['lhand_mesh']
#         hand_add_cam = bat['hand_add_cam']
#         hand_change = bat['hand_change']
#         face_bbox = bat['face_bbox'].tolist()
#         rhand_bbox = bat['rhand_bbox'].tolist()
#         lhand_bbox = bat['lhand_bbox'].tolist()
#
#         init_face_princpt = (cfg.input_face_img_shape[1]/2, cfg.input_face_img_shape[0]/2)
#         init_hand_princpt = (cfg.input_hand_img_shape[1]/2, cfg.input_hand_img_shape[0]/2)
#
#     ###################
#         ori_focal_face = make_focal(cfg.focal,cfg.input_face_img_shape,face_bbox)
#         ori_focal_rh = make_focal(cfg.focal,cfg.input_hand_img_shape,rhand_bbox)
#         ori_focal_lh = make_focal(cfg.focal,cfg.input_hand_img_shape,lhand_bbox)
#
#
#         ori_princpt_face = make_princpt(init_face_princpt,cfg.input_face_img_shape,face_bbox)
#         ori_princpt_rh = make_princpt(init_hand_princpt,cfg.input_hand_img_shape,rhand_bbox)
#         ori_princpt_lh = make_princpt(init_hand_princpt,cfg.input_hand_img_shape,lhand_bbox)
#
#         ori_face_fix_face = source_target_portion(ori_princpt_face, fix_princpt)
#
#         por_princpt_face = fix_princpt
#         por_princpt_rh = [ori_face_fix_face[0]+ori_princpt_rh[0],ori_face_fix_face[1]+ori_princpt_rh[1]]
#         por_princpt_lh = [ori_face_fix_face[0]+ori_princpt_lh[0],ori_face_fix_face[1]+ori_princpt_lh[1]]
#
#         face_rendered_img = render_mesh(vis_img, face_mesh, flame.face, {'focal': ori_focal_face, 'princpt': por_princpt_face, 'add_cam_trans': face_add_cam})
#         if hand_change != 0:
#             right_rendered_img = render_mesh(face_rendered_img, rhand_mesh, mano.face['right'],
#                                              {'focal': ori_focal_rh, 'princpt': por_princpt_rh,
#                                               'add_cam_trans': hand_add_cam[0].unsqueeze(0)})
#             final_rendered_img = render_mesh(right_rendered_img, lhand_mesh, mano.face['left'],
#                                             {'focal': ori_focal_lh, 'princpt': por_princpt_lh,
#                                              'add_cam_trans': hand_add_cam[1].unsqueeze(0)})
#         else:
#             left_rendered_img = render_mesh(face_rendered_img, lhand_mesh, mano.face['left'], {'focal': ori_focal_lh , 'princpt': por_princpt_lh, 'add_cam_trans': hand_add_cam[1].unsqueeze(0)})
#             final_rendered_img = render_mesh(left_rendered_img, rhand_mesh, mano.face['right'], {'focal': ori_focal_rh , 'princpt': por_princpt_rh,'add_cam_trans': hand_add_cam[0].unsqueeze(0)})
#
#         # final_rendered_img = final_rendered_img.astype(np.uint8).copy()
#         final_rendered_img = final_rendered_img.astype(np.uint8).copy()[:int(cfg.output_image_size[0]*0.7),:,:]
#         # cv2.imwrite(os.path.join(save_dir, 'render_original_img_{0:05d}.jpg').format(i), final_rendered_img)
#
#         # new_final_rendered_img = np.concatenate((final_rendered_img,substitle), axis=0)
#         cv2.imwrite(os.path.join(save_dir, 'render_original_img_{0:05d}.jpg').format(i), final_rendered_img)
#
#     except Exception as e:
#         print("error message :   ", e)
#         error_folder = os.path.join(cfg.error_log_save, _day)
#         os.makedirs(error_folder, exist_ok=True)
#         error_path = os.path.join(error_folder, "error_mesh.txt")
#         if not os.path.isfile(error_path):
#             file_txt = open(error_path, "w", encoding="UTF-8")
#             file_txt.close()
#         file_txt = open(error_path, "a", encoding="UTF-8")
#         file_txt.write(
#             "mesh_idx : {}, time : {}, --- error message : {}\n".format(i, datetime.now(KST), e))
#         file_txt.close()
#         continue
# t4=time.time()

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