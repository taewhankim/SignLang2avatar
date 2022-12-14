import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from main.config import cfg
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
import torch
from main.model import make_focal, make_princpt, source_target_portion
from common.utils.human_models import mano, flame
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
time_record = datetime.now(KST)
_day = str(time_record)[:10]
def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)

def render_mesh(img, mesh, face, cam_param):
    # mesh
    # mesh = (torch.tensor(mesh).to('cuda') - cam_param['add_cam_trans'])[0].detach().cpu().numpy()
    mesh = trimesh.Trimesh(mesh, face)

    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    # angle = -195
    # axis = [1, 0, 0]
    # R = trimesh.transformations.rotation_matrix(np.radians(angle), axis)
    # mesh.apply_transform(R)
    # mesh.vertices = (torch.tensor(mesh.vertices).to('cuda') + cam_param['add_cam_trans'])[0].detach().cpu().numpy()
    #
    # rot = trimesh.transformations.rotation_matrix(
    #     np.radians(180), [1, 0, 0])
    # mesh.apply_transform(rot)


    # material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(0.5, 0.5, 0.5, 1.0))
    # material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(119/255, 142/255, 190/255, 1.0))
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(124/255, 150/255, 210/255, 1.0))
    # material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(104/255, 131/255, 194/255, 1.0))


    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')


    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img = rgb * valid_mask + img * (1-valid_mask)
    # img = rgb * valid_mask
    return img

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def rendering_3d(i,bat,vis_img,fix_princpt):
    # try:
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
