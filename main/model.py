import torch
import torch.nn as nn
from torch.nn import functional as F
from common.nets.resnet import ResNetBackbone
from common.nets.module import PositionNet, RotationNet, FaceRegressor
from common.nets.loss import CoordLoss, ParamLoss
from common.utils.human_models import smpl, mano, flame
from common.utils.transforms import rot6d_to_axis_angle
from main.config import cfg
import math
import copy
from collections import Counter
import numpy as np
class Model(nn.Module):
    def __init__(self, networks, parts):
        super(Model, self).__init__()
        # body networks
        self.parts = parts
        if self.parts == 'body':
            self.backbone = networks['backbone']
            self.position_net = networks['position_net']
            self.rotation_net = networks['rotation_net']
            self.smpl_layer = copy.deepcopy(smpl.layer['neutral']).cuda()
            self.trainable_modules = [self.backbone, self.position_net, self.rotation_net]

        # hand networks
        elif self.parts == 'hand':
            self.backbone = networks['backbone']
            self.position_net = networks['position_net']
            self.rotation_net = networks['rotation_net']
            self.mano_layer = copy.deepcopy(mano.layer['right']).cuda()
            self.trainable_modules = [self.backbone, self.position_net, self.rotation_net]
        # face networks
        elif self.parts == 'face':
            self.backbone = networks['backbone']
            self.regressor = networks['regressor']
            self.flame_layer = copy.deepcopy(flame.layer).cuda()
            self.trainable_modules = [self.backbone, self.regressor]

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        
    def get_camera_trans(self, cam_param, input_img_shape):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        ## 수정
        # k_value = torch.FloatTensor([math.sqrt(cfg.focal[0]*cfg.focal[1]*cfg.camera_3d_size*cfg.camera_3d_size/(cfg.input_img_shape[0]*cfg.input_img_shape[1]))]).cuda().view(-1)
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0]*cfg.focal[1]*cfg.camera_3d_size*cfg.camera_3d_size/(input_img_shape[0]*input_img_shape[1]))]).cuda().view(-1)

        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def forward_position_net(self, inputs, backbone, position_net):
        img_feat = backbone(inputs['img'])
        joint_img = position_net(img_feat)
        return img_feat, joint_img
    
    def forward_rotation_net(self, img_feat, joint_img, rotation_net):
        batch_size = img_feat.shape[0]

        # parameter estimation
        if self.parts == 'body':
            root_pose_6d, pose_param_6d, shape_param, cam_param = rotation_net(img_feat, joint_img)
            # change 6d pose -> axis angles
            root_pose = rot6d_to_axis_angle(root_pose_6d)
            pose_param = rot6d_to_axis_angle(pose_param_6d.view(-1,6)).reshape(batch_size,-1)
            pose_param = torch.cat((pose_param, torch.zeros((batch_size,2*3)).cuda().float()),1) # add two zero hand poses
            cam_trans = self.get_camera_trans(cam_param)
            return root_pose, pose_param, shape_param, cam_trans

        elif self.parts == 'hand':
            root_pose_6d, pose_param_6d, shape_param, cam_param = rotation_net(img_feat, joint_img)
            # change 6d pose -> axis angles
            root_pose = rot6d_to_axis_angle(root_pose_6d).reshape(-1,3)
            pose_param = rot6d_to_axis_angle(pose_param_6d.view(-1,6)).reshape(-1,(mano.orig_joint_num-1)*3)
            cam_trans = self.get_camera_trans(cam_param,cfg.input_hand_img_shape)
            return root_pose, pose_param, shape_param, cam_trans

    def get_coord(self, params, mode):
        batch_size = params['root_pose'].shape[0]

        if self.parts == 'body':
            output = self.smpl_layer(global_orient=params['root_pose'], body_pose=params['body_pose'], betas=params['shape'])
            # camera-centered 3D coordinate
            mesh_cam = output.vertices
            joint_cam = torch.bmm(torch.from_numpy(smpl.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
            root_joint_idx = smpl.root_joint_idx
        elif self.parts == 'hand':
            output = self.mano_layer(global_orient=params['root_pose'], hand_pose=params['hand_pose'], betas=params['shape'])
            # camera-centered 3D coordinate
            mesh_cam = output.vertices
            joint_cam = torch.bmm(torch.from_numpy(mano.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
            root_joint_idx = mano.root_joint_idx
        elif self.parts == 'face':
            zero_pose = torch.zeros((1,3)).float().cuda().repeat(batch_size,1) # zero pose for eyes and neck
            output = self.flame_layer(global_orient=params['root_pose'], jaw_pose=params['jaw_pose'], betas=params['shape'], expression=params['expr'], neck_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose)
            # camera-centered 3D coordinate
            mesh_cam = output.vertices
            joint_cam = output.joints
            root_joint_idx = flame.root_joint_idx
        if mode == 'test' and cfg.testset == 'AGORA': # use 45 joints for AGORA evaluation
            joint_cam = output.joints

        # project 3D coordinates to 2D space
        cam_trans = params['cam_trans']
        if mode == 'train' and len(cfg.trainset_3d) == 1 and cfg.trainset_3d[0] == 'AGORA' and len(cfg.trainset_2d) == 0: # prevent gradients from backpropagating to SMPL/MANO/FLAME paraemter regression module
            x = (joint_cam[:,:,0].detach() + cam_trans[:,None,0]) / (joint_cam[:,:,2].detach() + cam_trans[:,None,2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:,:,1].detach() + cam_trans[:,None,1]) / (joint_cam[:,:,2].detach() + cam_trans[:,None,2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        else:
            x = (joint_cam[:,:,0] + cam_trans[:,None,0]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:,:,1] + cam_trans[:,None,1]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]


        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam

        # add camera translation for the rendering
        mesh_cam = mesh_cam + cam_trans[:,None,:]

        return joint_proj, joint_cam, mesh_cam, cam_trans[:,None,:]

    def batch_orth_proj(self,X, camera):
        '''
            X is N x num_verts x 3
        '''
        camera = camera.clone().view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
        Xn = (camera[:, :, 0:1] * X_trans)
        return Xn

    def forward(self, inputs, targets, meta_info,i,mode):
        # network forward and get outputs
        # body network
        if self.parts == 'body':
            img_feat, joint_img = self.forward_position_net(inputs, self.backbone, self.position_net)
            smpl_root_pose, smpl_body_pose, smpl_shape, cam_trans = self.forward_rotation_net(img_feat, joint_img.detach(), self.rotation_net)
            joint_proj, joint_cam, mesh_cam, add_cam_trans = self.get_coord({'root_pose': smpl_root_pose, 'body_pose': smpl_body_pose, 'shape': smpl_shape, 'cam_trans': cam_trans}, mode)
            smpl_body_pose = smpl_body_pose.view(-1,(smpl.orig_joint_num-1)*3)
            smpl_pose = torch.cat((smpl_root_pose, smpl_body_pose),1)

        # hand network
        elif self.parts == 'hand':
            img_feat, joint_img = self.forward_position_net(inputs, self.backbone, self.position_net)
            mano_root_pose, mano_hand_pose, mano_shape, cam_trans = self.forward_rotation_net(img_feat, joint_img.detach(), self.rotation_net)
            # if i == 0:
            #     cfg.first_hand_shape = mano_shape
            # else:
            #     mano_shape = cfg.first_hand_shape
            joint_proj, joint_cam, mesh_cam, add_cam_trans = self.get_coord({'root_pose': mano_root_pose, 'hand_pose': mano_hand_pose, 'shape': mano_shape, 'cam_trans': cam_trans}, mode)
            mano_hand_pose = mano_hand_pose.view(-1,(mano.orig_joint_num-1)*3)
            mano_pose = torch.cat((mano_root_pose, mano_hand_pose),1)

        # face network
        elif self.parts == 'face':
            img_feat = self.backbone(inputs['img'])
            flame_root_pose, flame_jaw_pose, flame_shape, flame_expr, cam_param = self.regressor(img_feat)
            flame_root_pose = rot6d_to_axis_angle(flame_root_pose)
            flame_jaw_pose = rot6d_to_axis_angle(flame_jaw_pose)
            cam_trans = self.get_camera_trans(cam_param, cfg.input_face_img_shape)
            if i == 0:
                cfg.first_face_shape = flame_shape
                # cfg.first_cam_trans = cam_trans
            else:
                flame_shape = cfg.first_face_shape
                # cam_trans = cfg.first_cam_trans
            joint_proj, joint_cam, mesh_cam , add_cam_trans= self.get_coord({'root_pose': flame_root_pose, 'jaw_pose': flame_jaw_pose, 'shape': flame_shape, 'expr': flame_expr, 'cam_trans': cam_trans}, mode)

        if mode == 'train':
            # loss functions
            loss = {}
            if cfg.parts == 'body':
                loss['joint_img'] = self.coord_loss(joint_img, smpl.reduce_joint_set(targets['joint_img']), smpl.reduce_joint_set(meta_info['joint_trunc']), meta_info['is_3D'])
                loss['smpl_joint_img'] = self.coord_loss(joint_img, smpl.reduce_joint_set(targets['smpl_joint_img']), smpl.reduce_joint_set(meta_info['smpl_joint_trunc']))
                loss['smpl_pose'] = self.param_loss(smpl_pose, targets['smpl_pose'], meta_info['smpl_pose_valid'])
                loss['smpl_shape'] = self.param_loss(smpl_shape, targets['smpl_shape'], meta_info['smpl_shape_valid'][:,None])
                loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:,:,:2], meta_info['joint_trunc'])
                loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None])
                loss['smpl_joint_cam'] = self.coord_loss(joint_cam, targets['smpl_joint_cam'], meta_info['smpl_joint_valid'])
                
            elif cfg.parts == 'hand':
                loss['joint_img'] = self.coord_loss(joint_img, targets['joint_img'], meta_info['joint_trunc'], meta_info['is_3D'])
                loss['mano_joint_img'] = self.coord_loss(joint_img, targets['mano_joint_img'], meta_info['mano_joint_trunc'])
                loss['mano_pose'] = self.param_loss(mano_pose, targets['mano_pose'], meta_info['mano_pose_valid'])
                loss['mano_shape'] = self.param_loss(mano_shape, targets['mano_shape'], meta_info['mano_shape_valid'][:,None])
                loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:,:,:2], meta_info['joint_trunc'])
                loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None])
                loss['mano_joint_cam'] = self.coord_loss(joint_cam, targets['mano_joint_cam'], meta_info['mano_joint_valid'])

            elif cfg.parts == 'face':
                loss['flame_root_pose'] = self.param_loss(flame_root_pose, targets['flame_root_pose'], meta_info['flame_root_pose_valid'][:,None])
                loss['flame_jaw_pose'] = self.param_loss(flame_jaw_pose, targets['flame_jaw_pose'], meta_info['flame_jaw_pose_valid'][:,None])
                loss['flame_shape'] = self.param_loss(flame_shape, targets['flame_shape'], meta_info['flame_shape_valid'][:,None])
                loss['flame_expr'] = self.param_loss(flame_expr, targets['flame_expr'], meta_info['flame_expr_valid'][:,None])
                loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:,:,:2], meta_info['joint_trunc'])
                loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None])
                loss['flame_joint_cam'] = self.coord_loss(joint_cam, targets['flame_joint_cam'], meta_info['flame_joint_valid'])

            return loss
        else:
            # test output
            out = {'cam_trans': cam_trans} 
            if self.parts == 'body':
                out['img'] = inputs['img']
                out['joint_img'] = joint_img
                out['smpl_mesh_cam'] = mesh_cam
                out['smpl_joint_proj'] = joint_proj
                out['smpl_pose'] = smpl_pose
                out['smpl_shape'] = smpl_shape
                if 'smpl_mesh_cam' in targets:
                    out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
                if 'bb2img_trans' in meta_info:
                    out['bb2img_trans'] = meta_info['bb2img_trans']
            elif self.parts == 'hand':
                out['img'] = inputs['img']
                out['joint_img'] = joint_img 
                out['mano_mesh_cam'] = mesh_cam
                out['mano_pose'] = mano_pose
                out['mano_shape'] = mano_shape
                out['add_cam_trans'] = add_cam_trans
                if 'mano_mesh_cam' in targets:
                    out['mano_mesh_cam_target'] = targets['mano_mesh_cam']
                if 'joint_img' in targets:
                    out['joint_img_target'] = targets['joint_img']
                if 'joint_valid' in meta_info:
                    out['joint_valid'] = meta_info['joint_valid']
                if 'bb2img_trans' in meta_info:
                    out['bb2img_trans'] = meta_info['bb2img_trans']
            elif self.parts == 'face':
                out['img'] = inputs['img']
                out['flame_joint_cam'] = joint_cam
                out['flame_mesh_cam'] = mesh_cam
                out['flame_root_pose'] = flame_root_pose
                out['flame_jaw_pose'] = flame_jaw_pose
                out['flame_shape'] = flame_shape
                out['flame_expr'] = flame_expr
                out['add_cam_trans'] = add_cam_trans

            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode,parts):
    if parts == 'body':
        backbone = ResNetBackbone(cfg.resnet_type)
        position_net = PositionNet()
        rotation_net = RotationNet()
        if mode == 'train':
            backbone.init_weights()
            position_net.apply(init_weights)
            rotation_net.apply(init_weights)
        model = Model({'backbone': backbone, 'position_net': position_net, 'rotation_net': rotation_net})
        return model

    if parts == 'hand':
        backbone = ResNetBackbone(cfg.resnet_type)
        position_net = PositionNet()
        rotation_net = RotationNet()
        if mode == 'train':
            backbone.init_weights()
            position_net.apply(init_weights)
            rotation_net.apply(init_weights)
        model = Model({'backbone': backbone, 'position_net': position_net, 'rotation_net': rotation_net},parts)
        return model

    if parts == 'face':
        backbone = ResNetBackbone(cfg.resnet_type)
        regressor = FaceRegressor()
        if mode == 'train':
            backbone.init_weights()
            regressor.apply(init_weights)
        model = Model({'backbone': backbone, 'regressor': regressor},parts)
        return model


def batch_orth_proj(X, camera):
    '''
        X is N x num_verts x 3
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn

def make_princpt(princpt,input_img,bbox, portion=None):
    if not portion:
        result = [(princpt[0] / input_img[1] * bbox[2] + bbox[0]),
                (princpt[1] / input_img[0] * bbox[3] + bbox[1])]
    else:
        result = [(princpt[0] / input_img[1] * bbox[2] + bbox[0]) * portion[0],
                (princpt[1] / input_img[0] * bbox[3] + bbox[1]) * portion[1]]
    return result

def make_focal(focal,input_img,bbox, portion=None):
    if not portion:
        result = [focal[0] / input_img[1] * bbox[2],
                 focal[1] / input_img[0] * bbox[3]]
    else:
        result = [focal[0] / input_img[1] * bbox[2] * portion[0],
                 focal[1] / input_img[0] * bbox[3] * portion[1]]
    return result


def source_target_portion(ori, target):
# def source_target_portion(ori, target, portion):
    # new_target = [target[0]/ori[0],target[1]/ori[1]]
    # new_target = [(target[0] - ori[0])*portion[0],(target[1] - ori[1])*portion[1]]
    new_target = [(target[0] - ori[0]),(target[1] - ori[1])]

    return new_target

def princpt_portion(portion, target):
    return [portion[0]*target[0],portion[1]*target[1]]


def bbox_portion_mod(bbox, portion):
    x1,y1,w,h = bbox
    new_x1,new_w = x1 * portion[0],w * portion[0]
    new_y1,new_h = y1 * portion[1],h * portion[1]
    new_bbox = new_x1,new_y1,new_w,new_h
    return new_bbox

def wh_portion_mod(bbox,portion):
    x1,y1,w,h = bbox
    c_x = x1+w/2
    c_y = y1+h/2
    new_w, new_h = w * portion[0], h * portion[1]
    new_x1 = c_x - new_w/2
    new_y1 = c_y - new_h/2
    return new_x1,new_y1,new_w,new_h

def cont_dict(args, cont, bin_list, result_list):
    bin_list.append(cont)
    zcnt=0
    if len(bin_list) != args.cont_n:
        pass
    else:
        c = Counter()
        for tmp in bin_list:
            if np.all(tmp['face_mesh']==0):
            # if tmp['face_mesh'].all() == 0:
                zcnt+=1
                c.update(tmp)
            else:
                if zcnt==0:
                    c.update(tmp)
                else:
                    tmp['face_mesh'] = tmp['face_mesh']*(zcnt+1)
                    tmp['face_add_cam'] = tmp['face_add_cam']*(zcnt+1)
                    tmp['face_bbox'] = tmp['face_bbox'] *(zcnt+1)
                    zcnt=0
                    c.update(tmp)
        for key, val in c.items():
            c[key] = val / args.cont_n
        result_list.append(c)
        bin_list.pop(0)
    return result_list


def split_frame_range(frame_cnt):
    split_frame = []
    sum_frame = list(range(sum(frame_cnt)))
    temp_loc = 0
    for i in range(len(frame_cnt)):
        split_frame.append(sum_frame[temp_loc : temp_loc + frame_cnt[i]])
        temp_loc += frame_cnt[i]
    return split_frame


# def find_zeros(zero_idx, split_frame_range, tmp_result):
#     for zero in zero_idx:
#         cnt=0
#         for i in split_frame_range:
#             if zero in i:
#                 cnt+=1
#                 new_tmp = tmp_result[i]
#                 c = Counter()
#                 tmp_result
#             else:
#                 pass
