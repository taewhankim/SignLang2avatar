from main.config import cfg
import os
from glob import glob
import natsort
import json

path = "/mnt/dms/HAND_SIGN/saved_val_video/frame/original"
tmp_dict = {'video_name' : {},
            'mesh' : {}}
for root,dir,file in os.walk(path):
    if len(dir)>1:
        dir = natsort.natsorted(dir)
        for i in dir:
            i = i.split("_")[0]
            r1 = os.path.join(root,i)
            imagepath_list = glob(r1 + '/*.jpg') + glob(r1 + '/*.png') + glob(r1 + '/*.jpeg')
            len_img = len(imagepath_list)
            tmp_dict['video_name'][i] = len_img

def make_json(new_json_dir, dict):
    with open(new_json_dir, 'w', encoding='utf-8') as make_file:
        json.dump(dict, make_file, ensure_ascii=False, indent='\t')


save_path = '/mnt/dms/HAND_SIGN/frame_cnt.json'
make_json(save_path,tmp_dict)

#
# def read_json(json_file):
#     with open(json_file, 'r') as tmp_json:
#         tmp = json.load(tmp_json)
#         video_names = tmp['video_name']
#         meshes = tmp['mesh']
#         return video_names, meshes
# load_json = '/mnt/dms/HAND_SIGN/frame_cnt.json'
# video_names, meshes = read_json(load_json)
# print(video_names)
# print(meshes)
#
