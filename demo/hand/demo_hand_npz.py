import copy
import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from datetime import datetime, timezone, timedelta

[sys.path.append(os.path.join(os.path.abspath('.'),i)) for i in ["main",'data','common']]

sys.path.insert(0, osp.join('..', '..', 'main'))
sys.path.insert(0, osp.join('..', '..', 'data'))
sys.path.insert(0, osp.join('..', '..', 'common'))
[sys.path.append(i) for i in ['.', '..']]
from main.config import cfg
from main.model import get_model , make_princpt,make_focal, cont_dict, bbox_portion_mod
from common.utils.preprocessing import process_bbox, generate_patch_image, bbox_preprocessing
from common.utils.human_models import smpl, smpl_x, mano, flame
from common.utils.vis import render_mesh, visualize_grid
from common.utils.renderer import *
from infer_pipe import Speech2HandsignFP
import json
from tqdm import tqdm
from data.dataset import TestData, make_json, read_json
from collections import Counter
import time
from PIL import ImageFont,ImageDraw, Image
from glob import glob
from collections import defaultdict
import natsort
import shutil
import gc
import warnings

os.environ['PYOPENGL_PLATFORM'] = 'egl'
warnings.filterwarnings(action='ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('-p', '--inputpath', default='', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('--input_text', default=None,type=str, help = "input hand_sign sentences")
    parser.add_argument('-s', '--savefolder', default='', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='set device, cpu for using cpu')
    parser.add_argument('--rotation', default=False, action='store_true', help='rotation 270degree')
    parser.add_argument('--cont_n', default=1, type=int,
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

## 시간
KST = timezone(timedelta(hours=9))
time_record = datetime.now(KST)
_day = str(time_record)[:10]

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

if args.input_text:
    hand_sign_s2s = Speech2HandsignFP(vocab_path=cfg.hand_sign_vocab_path, ckpt_path=cfg.hand_sign_ckpt_path,
                                      handsign_folder_path=cfg.handsign_folder_path)
    hand_sign_output = hand_sign_s2s.translate(args.input_text)
    print("########## ORIGINAL COMMENTS    :   ", args.input_text, "     ##########")
    print("########## HAND SIGN TRANSLATIONS    :   ", hand_sign_output[0], "     ##########")

# save_obj_dir = os.path.join(args.savefolder, 'obj')
# os.makedirs(save_obj_dir, exist_ok=True)



# save_json = os.path.join(args.savefolder, 'param_json')
# os.makedirs(save_json, exist_ok=True)



t2 =time.time()
print("preprocess time :",t2-t1)

## video list
if os.path.isdir(args.inputpath):
    videopath_list = glob(args.inputpath +'/*.mp4') + glob(args.inputpath + '/*.mkv') + glob(args.inputpath +'/*.MOV')
    videopath_list = natsort.natsorted(videopath_list)
else:
    videopath_list = [args.inputpath]

# frame_cnt_dir = read_json(cfg.frame_cnt_dir)
for ida, vid_list in enumerate(tqdm(videopath_list,dynamic_ncols=True)):
    tmp_dict = defaultdict()
    tmp = []
    tmp_result = []
    # each_original_img_list = []
    anno_list = []
    testdata = TestData(vid_list, args.rotation, args.savefolder, device='cuda')
    video_name = vid_list.split('/')[-1].split('.')[0]

    save_render_folder = testdata.render_image_save_folder
    save_video_folder = os.path.join(args.savefolder,'video')
    os.makedirs(save_video_folder, exist_ok=True)

    mesh_info_dir = testdata.mesh_json_save_folder

    in_dir2 = save_render_folder
    out_path2 = os.path.join(save_video_folder, "{}_result.mp4".format(video_name))
    videonames = [{}]
    cnt=0

    frame_name = []
    result_data_list = []
    mesh_info = {
        "frame_name": frame_name,
        "original_image_size": None,
        "output_image_size": cfg.output_image_size,
        "mesh_info": None
    }

    try :
        for i, batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
            try:
                original_img = batch['original_img']
                # each_original_img_list.append(batch['original_img'])
                rhand_bbox, lhand_bbox = batch['right_hand'], batch['left_hand']
                hand_change = batch['hand_change']

                mesh_info['original_image_size'] = original_img.shape[:2]
                # mesh_info ={ cnt : {
                #                      "frame_name" : batch['image_name']+'.jpg',
                #                      "original_image_size" : original_img.shape[:2],
                #                      "output_image_size" : cfg.output_image_size,
                #                      "mesh_info" : None
                #                   }
                #             }



                # mesh_info = {"idx":i,
                #              "frame_img_name" : batch['image_name']+'.jpg',
                #              "original_image_size" : original_img.shape[:2],
                #              "output_image_size" : None,
                #              "mesh_info" : None}

                # # prepare input image
                transform = transforms.ToTensor()
                original_img_height, original_img_width = original_img.shape[:2]

                # rhand_bbox = process_bbox(rhand_bbox, original_img_width, original_img_height)
                # rhand_bbox = list(map(int,rhand_bbox))
                #
                # rhand_img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, rhand_bbox, 1.0, 0.0, False, cfg.input_hand_img_shape)
                # rhand_img = transform(rhand_img.astype(np.float32))/255
                # rhand_img = rhand_img.cuda()[None,:,:,:]
                # before_frame_rhand_img = copy.deepcopy(rhand_img)
                #
                # # prepare bbox (left hand)
                # lhand_bbox = process_bbox(lhand_bbox, original_img_width, original_img_height)
                # lhand_bbox = list(map(int,lhand_bbox)) # xmin, ymin, width, height
                #
                # ## 왼손 반전
                # lhand_img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, lhand_bbox, 1.0, 0.0, True, cfg.input_hand_img_shape) # flip to the right hand image
                # lhand_img = transform(lhand_img.astype(np.float32))/255
                # lhand_img = lhand_img.cuda()[None,:,:,:]
                # before_frame_lhand_img = copy.deepcopy(lhand_img)


                portion_h = cfg.output_image_size[0] / original_img.shape[0]
                portion_w = cfg.output_image_size[1] / original_img.shape[1]
                portion = [portion_w, portion_h]
                targets = {}
                meta_info = {}
                ## face prepro
                if batch['face'] is None:
                    pass
                else:
                    face_bbox = batch['face']
                    # face_bbox = face_bbox[0],face_bbox[1], face_bbox[2]-face_bbox[0], face_bbox[3]-face_bbox[1]
                    face_bbox = list(map(int,face_bbox))

                    face_img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, face_bbox, 1.0, 0.0, False, cfg.input_face_img_shape)
                    face_img = transform(face_img.astype(np.float32)) / 255
                    face_img = face_img.cuda()[None, :, :, :]
                    face_inputs = {'img': face_img}
                    with torch.no_grad():
                        face_out = face_model(face_inputs, targets, meta_info,cnt, 'test')
                    ## face output ## 수정 : add cam cpu화 시킴
                    face_add_cam = face_out['add_cam_trans'].detach().cpu().numpy()[0]
                    face_mesh = face_out['flame_mesh_cam'].detach().cpu().numpy()[0]
                    face_bbox = bbox_portion_mod(face_bbox, portion)
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
                    out = hand_model(inputs, targets, meta_info,cnt, 'test')

                ## hand output ## .detach().cpu().numpy()[0] 수정
                hand_add_cam = out['add_cam_trans'].detach().cpu().numpy()

                rhand_mesh = out['mano_mesh_cam'].detach().cpu().numpy()[0]
                lhand_mesh = out['mano_mesh_cam'].detach().cpu().numpy()[1]
                lhand_mesh[:, 0] *= -1  # flip back to the left hand mesh

                ## 추가
                ori_rh = copy.deepcopy(rhand_bbox)
                ori_lh = copy.deepcopy(lhand_bbox)
                ##

                rhand_bbox = bbox_portion_mod(rhand_bbox,portion)
                lhand_bbox = bbox_portion_mod(lhand_bbox,portion)

                n_rhand_bbox = copy.deepcopy(np.array(rhand_bbox))
                n_lhand_bbox = copy.deepcopy(np.array(lhand_bbox))

                # face_mesh_list.append(face_mesh)
                # face_add_cam_list.append(face_add_cam)
                # face_bbox_list.append(n_face_bbox)
                # rhand_mesh_list.append(rhand_mesh)
                # lhand_mesh_list.append(lhand_mesh)
                # hand_add_cam_list.append(hand_add_cam)
                # rhand_bbox_list.append(n_rhand_bbox)
                # lhand_bbox_list.append(n_lhand_bbox)
                # hand_change_list.append(hand_change)
                result_data = {"face_mesh": face_mesh,
                               "face_add_cam":face_add_cam,
                               "face_bbox": n_face_bbox,
                               "rhand_mesh": rhand_mesh,
                               "lhand_mesh": lhand_mesh,
                               "hand_add_cam": hand_add_cam,
                               "rhand_bbox":n_rhand_bbox,
                               "lhand_bbox":n_lhand_bbox,
                               "hand_change" : hand_change
                               }
                frame_name.append(batch['image_name']+'.jpg')
                result_data_c = copy.deepcopy(result_data)
                result_data_list.append(result_data_c)
                # tmp_result,check = cont_dict(args,result_data,tmp,tmp_result)
                # result_data2 = copy.deepcopy(result_data)
                # for tor_k,tor_v in result_data2.items():
                #     if isinstance(tor_v,np.ndarray):
                #         result_data2[tor_k] = tor_v.tolist()
                # mesh_info[cnt]['mesh_info'] = result_data2

                # anno_list.append(mesh_info)

                cnt+=1
                # del mesh_info
                # del rhand_bbox
                # del lhand_bbox
                del out
                del rhand_mesh
                del lhand_mesh
                del result_data
                del result_data_c
                gc.collect()
            except Exception as e:
                error_path = os.path.join(args.savefolder, "error_video_val.txt")
                KST = timezone(timedelta(hours=9))
                if not os.path.isfile(error_path):
                    file_txt = open(error_path, "w", encoding="UTF-8")
                    file_txt.close()
                file_txt = open(error_path, "a", encoding="UTF-8")
                file_txt.write(
                    "video_list : {}, image_name : {}, time : {}, --- error message : {}\n".format(video_name, batch[
                        'image_name'] + '.jpg', datetime.now(KST), e))
                pass

        # mesh_info['mesh_info'] = {
        #     "face_mesh": face_mesh_list,
        #     "face_add_cam": face_add_cam_list,
        #     "face_bbox": face_bbox_list,
        #     "rhand_mesh": rhand_mesh_list,
        #     "lhand_mesh": lhand_mesh_list,
        #     "hand_add_cam": hand_add_cam_list,
        #     "rhand_bbox": rhand_bbox_list,
        #     "lhand_bbox": lhand_bbox_list,
        #     "hand_change": hand_change_list
        # }
        mesh_info['mesh_info'] = result_data_list
        # videonames[0]['video_name'] = vid_list.split('/')[-1]
        # videonames[0]['frame_cnt'] = cnt
        videonames = {}
        videonames['video_name'] = vid_list.split('/')[-1]
        videonames['frame_cnt'] = cnt
        if os.path.isfile(os.path.join(mesh_info_dir, video_name+".npz")):
            os.remove(os.path.join(mesh_info_dir, video_name+".npz"))
            print("remove remain json! and make new json~~~!!!")

        # total_json ={"video":videonames, 'mesh':mesh_info}
        # new_total = total_json.items()
        # elemen = list(new_total)
        # new_arr = np.array(elemen)
        np.savez(os.path.join(mesh_info_dir, video_name+".npz"),video=videonames, mesh=mesh_info)
        # make_json(os.path.join(mesh_info_dir, video_name+".json"), video = videonames, mesh = mesh_info)
        # del anno_list
        del mesh_info
        del result_data_list
        gc.collect()

        # if os.path.isfile(args.inputpath) and (args.inputpath[-3:] in ['mp4', 'csv', 'MOV']):
        #     vidcap= cv2.VideoCapture(args.inputpath)
        #     fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        #     # frame_width = round(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     # frame_height = round(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     frame_width = 1080
        #     frame_height = 650
        # else:
    #     fps = 60
    #     frame_width = cfg.output_image_size[1]
    #     frame_height = int(cfg.output_image_size[1]*0.56)+150 #1080
    #
    #     # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     # outcap_only_hand = cv2.VideoWriter('{}/{}.mp4'.format(args.savefolder, "only_hand"),
    #     #                                fourcc, fps, (frame_height, frame_width))
    #
    #
    #     vis_img = np.ones([cfg.output_image_size[0],cfg.output_image_size[1], 3]) * 255
    #     ## for substitle
    #     # substitle = np.ones([150,cfg.output_image_size[1],3],np.uint8)*255
    #     # substitle2 = Image.fromarray(substitle)
    #     # draw = ImageDraw.Draw(substitle2)
    #     # fonts = ImageFont.truetype(cfg.font_dir,50)
    #     # f_w, f_h = fonts.getsize(args.input_text)
    #     # draw.text((int((substitle.shape[1]-f_w)/2),(int(substitle.shape[0]-f_h)/2) ), args.input_text, font=fonts, fill=(0,0,0))
    #     # substitle = np.array(substitle2)
    #
    #
    #     fix_princpt = [int(cfg.output_image_size[1]*0.5), int(cfg.output_image_size[0]*0.19)]
    #
    #     ## ready for grid original image
    #
    #     total_frame_length = len(each_original_img_list)
    #     contract = len(tmp_result)
    #     trim_size = total_frame_length - contract
    #     for_trim = trim_size // 2
    #     backward_trim = trim_size - for_trim
    #     total_batch = each_original_img_list[for_trim:-backward_trim]
    #
    #
    #     #####
    #     t3 = time.time()
    #     print("model time : ",t3-t2)
    #     print("##########################    mesh rendering    ##########################")
    #     for i , (bat,total_bat) in enumerate(tqdm(zip(tmp_result,total_batch), dynamic_ncols=True)):
    #         # vis_img = original_img.copy()
    #         vis_img = np.ones([vis_img.shape[0],vis_img.shape[1],vis_img.shape[2]])*255
    #         face_mesh = bat['face_mesh']
    #         face_mesh = bat['face_mesh']
    #         face_add_cam = torch.tensor(bat['face_add_cam']).to("cuda")
    #         rhand_mesh = bat['rhand_mesh']
    #         lhand_mesh = bat['lhand_mesh']
    #         hand_add_cam = torch.tensor(bat['hand_add_cam']).to("cuda")
    #         face_bbox = bat['face_bbox'].tolist()
    #         rhand_bbox = bat['rhand_bbox'].tolist()
    #         lhand_bbox = bat['lhand_bbox'].tolist()
    #
    #         hand_change = bat['hand_change']
    #
    #         init_face_princpt = (cfg.input_face_img_shape[1]/2, cfg.input_face_img_shape[0]/2)
    #         init_hand_princpt = (cfg.input_hand_img_shape[1]/2, cfg.input_hand_img_shape[0]/2)
    #
    #     ###################
    #         ori_focal_face = make_focal(cfg.focal,cfg.input_face_img_shape,face_bbox)
    #         ori_focal_rh = make_focal(cfg.focal,cfg.input_hand_img_shape,rhand_bbox)
    #         ori_focal_lh = make_focal(cfg.focal,cfg.input_hand_img_shape,lhand_bbox)
    #
    #         ori_princpt_face = make_princpt(init_face_princpt,cfg.input_face_img_shape,face_bbox)
    #         ori_princpt_rh = make_princpt(init_hand_princpt,cfg.input_hand_img_shape,rhand_bbox)
    #         ori_princpt_lh = make_princpt(init_hand_princpt,cfg.input_hand_img_shape,lhand_bbox)
    #
    #         face_rendered_img = render_mesh(vis_img, face_mesh, flame.face, {'focal': ori_focal_face, 'princpt': ori_princpt_face, 'add_cam_trans': face_add_cam})
    #
    #         if hand_change != 0:
    #             right_rendered_img = render_mesh(face_rendered_img, rhand_mesh, mano.face['right'],
    #                                          {'focal': ori_focal_rh, 'princpt': ori_princpt_rh,
    #                                           'add_cam_trans': hand_add_cam[0].unsqueeze(0)})
    #             final_rendered_img = render_mesh(right_rendered_img, lhand_mesh, mano.face['left'],
    #                                         {'focal': ori_focal_lh, 'princpt': ori_princpt_lh,
    #                                          'add_cam_trans': hand_add_cam[1].unsqueeze(0)})
    #         else:
    #             left_rendered_img = render_mesh(face_rendered_img, lhand_mesh, mano.face['left'], {'focal': ori_focal_lh , 'princpt': ori_princpt_lh, 'add_cam_trans': hand_add_cam[1].unsqueeze(0)})
    #             final_rendered_img = render_mesh(left_rendered_img, rhand_mesh, mano.face['right'], {'focal': ori_focal_rh , 'princpt': ori_princpt_rh,'add_cam_trans': hand_add_cam[0].unsqueeze(0)})
    #
    #
    #         final_rendered_img = final_rendered_img.astype(np.uint8).copy()
    #         # final_rendered_img = final_rendered_img.astype(np.uint8).copy()[:int(cfg.output_image_size[0]*0.56),:,:]
    #         # cv2.imwrite(os.path.join(save_dir, 'render_original_img_{0:05d}.jpg').format(i), final_rendered_img)
    #
    #         # new_final_rendered_img = np.concatenate((final_rendered_img,substitle), axis=0)
    #         ## for grid
    #         new_output = cv2.resize(final_rendered_img,(total_bat.shape[0:2]),interpolation = cv2.INTER_AREA)
    #         new_final_rendered_img = np.concatenate((total_bat,new_output), axis=1)
    #
    #         cv2.imwrite(os.path.join(save_render_folder, 'render_original_img_{0:05d}.jpg').format(i), new_final_rendered_img)
    #
    #         del init_face_princpt
    #         del init_hand_princpt
    #
    #         ## make video
    #         #outcap_only_hand.write(final_rendered_img)
    #     #outcap_only_hand.release()
    #     t4=time.time()
    #
    #     print("rendering time : ",t4-t3)
    #     print("Total time : ",t4-t1)
    #     ffmpeg_cmd2 = f'ffmpeg -y -f image2 -framerate 60 -pattern_type glob -i "{in_dir2}/*.jpg"  -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {out_path2}'
    #     os.system(ffmpeg_cmd2)
    #     del tmp_result
    #     del total_batch
    #     gc.collect()
    #
    except Exception as e:
        print("\n ################### error message :   ", e,"     #######################")
        print("###########################################################################\n")
        error_path = os.path.join(args.savefolder, "error_video_val.txt")
        KST = timezone(timedelta(hours=9))
        if not os.path.isfile(error_path):
            file_txt = open(error_path, "w", encoding="UTF-8")
            file_txt.close()
        file_txt = open(error_path, "a", encoding="UTF-8")
        file_txt.write("video_list : {}, image_name : {}, time : {}, --- error message : {}\n".format(video_name, batch['image_name']+'.jpg',datetime.now(KST), e))
        file_txt.close()


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
