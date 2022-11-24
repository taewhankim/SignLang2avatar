import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from models.synergynet.ddfa import ToTensor, Normalize
from models.synergynet.model_building import SynergyNet
from models.synergynet.inference import crop_img, predict_denseVert, predict_sparseVert, draw_landmarks
import argparse
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import os
import os.path as osp
import glob
# from FaceBoxes import FaceBoxes
from models.synergynet.render import render
import natsort

# Following 3DDFA-V2, we also use 120x120 resolution
IMG_SIZE = 120

class sy_net(nn.Module):
    def __init__(self):
        super(sy_net, self).__init__()
        self.checkpoint_fp = '/mnt/dms/KTW/synergynet/weights/best.pth.tar'
        self.arch = 'mobilenet_v2'
        # self.args.devices_id = [0]

        checkpoint = torch.load(self.checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        self.img_size = 120
        self.model = SynergyNet(self.img_size,self.arch)
        model_dict = self.model.state_dict()
        self.uv_vert=np.load('/mnt/dms/KTW/synergynet/weights/3dmm_data/BFM_UV.npy')
        self.coord_u = (self.uv_vert[:,1]*255.0).astype(np.int32)
        self.coord_v = (self.uv_vert[:,0]*255.0).astype(np.int32)
        self.keep_ind = np.load('/mnt/dms/KTW/synergynet/weights/3dmm_data/keptInd.npy')
        self.tri_deletion = np.load('/mnt/dms/KTW/synergynet/weights/3dmm_data/deletedTri.npy')

        # because the model is trained by multiple gpus, prefix 'module' should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]

        self.model.load_state_dict(model_dict, strict=False)
        self.model = self.model.cuda()
        self.model.eval()
        self.transform = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])

        # parser = argparse.ArgumentParser()
        # parser.add_argument('-f', '--files', default='', help='path to a single image or path to a folder containing multiple images')
        # parser.add_argument("--png", action="store_true", help="if images are with .png extension")
        # parser.add_argument('--img_size', default=120, type=int)
        # parser.add_argument('-b', '--batch-size', default=1, type=int)

    def write_obj_with_colors(self,obj_name, vertices, triangles, colors):
        triangles = triangles.copy()

        if obj_name.split('.')[-1] != 'obj':
            obj_name = obj_name + '.obj'
        with open(obj_name, 'w') as f:
            for i in range(vertices.shape[1]):
                s = 'v {:.4f} {:.4f} {:.4f} {} {} {}\n'.format(vertices[0, i], vertices[1, i], vertices[2, i], colors[i, 2],
                                                   colors[i, 1], colors[i, 0])
                f.write(s)
            for i in range(triangles.shape[1]):
                s = 'f {} {} {}\n'.format(triangles[0, i], triangles[1, i], triangles[2, i])
                f.write(s)

    def forward(self, img,face_size):
        # load pre-tained model
        # checkpoint_fp = '/mnt/dms/KTW/synergynet/weights/best.pth.tar'
        # arch = 'mobilenet_v2'
        # devices_id = [0]
        #
        # checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        #
        # model = SynergyNet(args)
        # model_dict = model.state_dict()
        #
        # # load BFM_UV mapping and kept indicies and deleted triangles
        # uv_vert=np.load('3dmm_data/BFM_UV.npy')
        # coord_u = (uv_vert[:,1]*255.0).astype(np.int32)
        # coord_v = (uv_vert[:,0]*255.0).astype(np.int32)
        # keep_ind = np.load('3dmm_data/keptInd.npy')
        # tri_deletion = np.load('3dmm_data/deletedTri.npy')
        #
        # # because the model is trained by multiple gpus, prefix 'module' should be removed
        # for k in checkpoint.keys():
        #     model_dict[k.replace('module.', '')] = checkpoint[k]
        #
        # model.load_state_dict(model_dict, strict=False)
        # model = model.cuda()
        # model.eval()

        # face detector
        # face_boxes = FaceBoxes()

        # preparation
        # if osp.isdir(args.files):
        #     if not args.files[-1] == '/':
        #         args.files = args.files + '/'
        #     if not args.png:
        #         files = sorted(glob.glob(args.files+'*.jpg'))
        #     else:
        #         files = sorted(glob.glob(args.files+'*.png'))
        # else:
        #     files = [args.files]

        # files = natsort.natsorted(files)
        # for img_fp in files:
        # print("Process the image: ", img_fp)

        img_ori = img.astype(np.uint8).copy()

        # crop faces
        rect = [0,0,256,256,1.0] # pre-cropped faces

        # storage
        vertices_lst = []
        roi_box = rect
        img = crop_img(img_ori, roi_box)
        img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        input = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            input = input.cuda()
            param = self.model.forward_test(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # dense pts
        # lmks = predict_sparseVert(param, roi_box, transform=True)
        # pts = []
        # pts.append(lmks)
        # draw_landmarks(img_ori, pts, wfp='/mnt/dms/KTW/hand4whole/results/새롭다2/crop_head/lmk_.jpg')

        vertices = predict_denseVert(param, roi_box, transform=True)
        vertices = vertices[:,self.keep_ind]
        vertices_lst.append(vertices)

        # textured obj file output
        # if not osp.exists(f'inference_output/obj/'):
        #     os.makedirs(f'inference_output/obj/')
        # if not osp.exists(f'inference_output/rendering_overlay/'):
        #     os.makedirs(f'inference_output/rendering_overlay/')

        # name = img_fp.rsplit('/',1)[-1][:-11] # drop off the postfix
        # colors = cv2.imread(f'texture_data/uv_real/{name}_fake_B.png',-1)
        # colors = cv2.imread('/mnt/dms/KTW/synergynet/weights/image00874_fake_B.png',-1)
        colors = cv2.imread('/mnt/dms/KTW/synergynet/weights/image00874_output_out.png',-1)

        colors = np.flip(colors,axis=0)
        colors_uv = (colors[self.coord_u, self.coord_v,:])

        # wfp = f'inference_output/obj/{name}.obj'
        # write_obj_with_colors(wfp, vertices, self.tri_deletion, colors_uv[self.keep_ind,:].astype(np.float32))

        tex = colors_uv[self.keep_ind,:].astype(np.float32)/255.0
        # render(img_ori, vertices_lst, alpha=0.6, wfp=f'/mnt/dms/KTW/synergynet/results/3/{cnt}.jpg', tex=tex, connectivity=self.tri_deletion-1)
        result_img = render(img_ori, vertices_lst, face_size, alpha=0.6, wfp=None, tex=tex, connectivity=self.tri_deletion-1)
        return result_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', default='', help='path to a single image or path to a folder containing multiple images')
    parser.add_argument("--png", action="store_true", help="if images are with .png extension")
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int)

    args = parser.parse_args()
    # main(args)