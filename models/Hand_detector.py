import sys
import torch
import numpy as np
import os

sys.path.append(os.path.join(os.path.abspath('.')))

from detectors.hand_detector_d import wrist_detect
from main.config import cfg as cfg2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from projects.HandLer.handler.config import add_handler_config
from projects.HandLer.tracking_utils import *
import cv2

from data import dataset

import warnings
warnings.filterwarnings(action='ignore')

class Hand_track_Detector(wrist_detect):
    """
    Hand Detector for third-view input.
    It combines a body pose estimator (https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git)
    with a type-agnostic hand detector (https://github.com/ddshan/hand_detector.d2)
    """

    def __init__(self,model_path, device):
        super(Hand_track_Detector, self).__init__()
        print("Loading Third View Hand Detector")
        self.confidence_threshold = 0.5
        self.IOU_threshold = 0.01
        self.retention_threshold = 20
        self.init_threshold = 0.8
        self.max_id = 0

        self.device = device
        self.__load_hand_detector(model_path,self.device)
        self.det_list_all = []
        self.tracklet_all = []
        self.tmp = dict()
        self.bin = []

    def __load_hand_detector(self,model_path,device):
        # load cfg and models
        cfg = get_cfg()
        add_handler_config(cfg)
        cfg.merge_from_file(cfg2.hand_config)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
        print("load weights from " + cfg.MODEL.WEIGHTS)
        self.hand_detector = CustomPredictor(cfg, device)


    def __get_raw_hand_bbox(self, img,index):
        bbox = self.run_each_dataset(self.hand_detector,img,index)
        bboxes = np.array(bbox)

        return bboxes

    def run_each_dataset(self,model, img, index):
        self.det_list_all.append([])
        image = img
        i = index
        bbox_result = []
        with torch.no_grad():
            if index == 0:
                outputs, self.heatmap, trk_res, self.feats = model({"image": image, "pre_image": image, "pose_hm": None})
            else:
                outputs, self.heatmap, trk_res, _ = model({"image": image, "pre_image": self.pre_img, "heatmap": self.heatmap, "pose_hm": None, 'pre_features': self.feats})

            trk_res = trk_res[0]
            outputs, trk_res = outputs[0], trk_res[0]
            bboxes = copy.deepcopy(outputs["instances"].pred_boxes.tensor.cpu().numpy())
            scores = copy.deepcopy(outputs["instances"].scores.cpu().numpy())
            pre_bboxes = copy.deepcopy(trk_res["instances"].pred_boxes.tensor.cpu().numpy())
            pre_scores = trk_res["instances"].scores.cpu().numpy()
            if len(pre_bboxes)>0 and len(pre_bboxes)==len(bboxes):
                self.pre_bboxes_with_scores = np.concatenate((pre_bboxes[None,:,:], pre_scores[None,:, None]), 2)
            elif len(bboxes)>0:
                self.pre_bboxes_with_scores = np.concatenate((bboxes, scores[None,:]), 1)
            self.pre_img = image


            length = bboxes.shape[0]
            for j in range(length):
                det_rect = set_det_rect(bboxes[j], scores[j], self.pre_bboxes_with_scores[:,j,:], index)
                if det_rect.conf > self.confidence_threshold:
                    self.det_list_all[det_rect.curr_frame].append(det_rect)

            if i == 0:
                for j in range(len(self.det_list_all[i])):
                    self.det_list_all[i][j].id = j + 1
                    self.max_id = max(self.max_id, j + 1)
                    track = tracklet(self.det_list_all[i][j])
                    self.tracklet_all.append(track)

            else:

                matches, unmatched1, unmatched2 = track_det_match(self.tracklet_all, self.det_list_all[i], self.IOU_threshold)

                matched_id_list = []
                for j in range(len(matches)):
                    self.det_list_all[i][matches[j][1]].id = self.tracklet_all[matches[j][0]].id
                    self.tracklet_all[matches[j][0]].add_rect(self.det_list_all[i][matches[j][1]])
                    matched_id_list.append(self.det_list_all[i][matches[j][1]].id)
                    continue



                for j in range(len(unmatched2)):
                    if self.det_list_all[i][unmatched2[j]].conf >= self.init_threshold:
                        self.det_list_all[i][unmatched2[j]].id = self.max_id + 1
                        self.max_id = self.max_id + 1
                        track = tracklet(self.det_list_all[i][unmatched2[j]])
                        self.tracklet_all.append(track)

                delete_track_list = []
                for j in range(len(unmatched1)):
                    self.tracklet_all[unmatched1[j]].no_match_frame = self.tracklet_all[unmatched1[j]].no_match_frame + 1
                    if (self.tracklet_all[unmatched1[j]].no_match_frame >= self.retention_threshold):
                        delete_track_list.append(unmatched1[j])

                origin_index = set([k for k in range(len(self.tracklet_all))])
                delete_index = set(delete_track_list)
                left_index = list(origin_index - delete_index)
                self.tracklet_all = [self.tracklet_all[k] for k in left_index]
    # **************visualize tracking result**************  **
        for j in range(len(self.det_list_all[-1])):
            x1, y1, x2, y2 = self.det_list_all[i][j].curr_rect.astype(int)
            trace_id = self.det_list_all[i][j].id
            if trace_id == 0:
                continue
            bbox_result.append([x1, y1, x2, y2,trace_id])
        return bbox_result
             
          

    #
    # def detect_hand_bbox(self, img, index):
    #     '''
    #         output:
    #             body_bbox: [min_x, min_y, width, height]
    #             hand_bbox: [x0, y0, x1, y1]
    #         Note:
    #             len(body_bbox) == len(hand_bbox), where hand_bbox can be None if not valid
    #     '''
    #     # get body pose
    #     body_pose_list, body_bbox_list = self.detect_body_pose(img)
    #     # assert len(body_pose_list) == 1, "Current version only supports one person"
    #
    #     # get raw hand bboxes
    #     raw_hand_bboxes = self.__get_raw_hand_bbox(img, index)
    #     ## original method
    #     # hand_bbox_list, raw_hand_bboxes = self.detect_hand_bbox_ori(body_pose_list, raw_hand_bboxes,body_bbox_list)
    #     if index==0:
    #         predict_bbox = self.detect_hand_bbox_openpose(img, body_pose_list)
    #         hand_bbox_list = self.classify_which_hand(raw_hand_bboxes, predict_bbox)
    #
    #         for k,v in hand_bbox_list[0].items():
    #             self.tmp[v[-1]] = k
    #             hand_bbox_list[0][k] = v[:-1]
    #         if len(self.tmp)!=2:
    #             right, left = self.simple_dist(raw_hand_bboxes)
    #             hand_bbox_list[0]['right_hand'] = right[:-1]
    #             hand_bbox_list[0]['left_hand'] = left[:-1]
    #             self.tmp[right[-1]] = 'right_hand'
    #             self.tmp[left[-1]] = 'left_hand'
    #         return hand_bbox_list
    #
    #     hand_bboxes = dict(
    #         left_hand=None,
    #         right_hand=None
    #     )
    #
    #     for idx, hands in enumerate(raw_hand_bboxes):
    #         hand = self.tmp[hands[-1]]
    #         hand_bboxes[hand] = hands[:-1]
    #     hand_bbox_list = [hand_bboxes]
    #     return hand_bbox_list

    def detect_hand_bbox(self, img, index):

        # get raw hand bboxes
        raw_hand_bboxes = self.__get_raw_hand_bbox(img, index)
        class_id = [i[-1] for i in raw_hand_bboxes]
        if index==0:
            # get wrist xy [left, right]
            wrist = self.detect_wrist(img)
            wrist_bbox = self.detect_hand_bbox_openpose(img,wrist)
            hand_bbox_list = self.classify_which_hand(raw_hand_bboxes, wrist_bbox)
            for k,v in hand_bbox_list[0].items():
                self.bin.append(v[-1])
                self.tmp[v[-1]] = k
                hand_bbox_list[0][k] = v[:-1]
            if (len(self.tmp)!=2) & (len(raw_hand_bboxes)!=1):
                if np.all(hand_bbox_list[0]['right_hand']==hand_bbox_list[0]['left_hand']):
                    c_bbox = hand_bbox_list[0]['right_hand']
                    for rh in raw_hand_bboxes:
                        if np.all(rh[:-1] == c_bbox):
                            pass
                        else:
                            rest_bbox = rh
                            break
                    hand_bbox_list, self.bin, self.tmp = self.simple_dist(rest_bbox, wrist, hand_bbox_list,self.bin)
                else:
                    pass
            return hand_bbox_list
        hand_bboxes = dict(
            left_hand=None,
            right_hand=None
        )
        ## check another id
        diff_id = list(set(self.bin) - set(class_id))
        for idx, hands in enumerate(raw_hand_bboxes):
            if (hands[-1] not in self.bin) & (len(diff_id)!=0): ## 기존의 id 중 하나가 새로운 id로 잡힌 경우
                hand = self.tmp[diff_id[0]]
                hand_bboxes[hand] = hands[:-1]
            elif hands[-1] not in self.bin: ## 기존의 id가 모두 존재하는데, raw_hands가 3개로 잡혀 idx 더 생긴 경우
                pass
            elif hands[-1] in self.bin: ## raw id와 기존 id가 동일한 경우
                hand = self.tmp[hands[-1]]
                hand_bboxes[hand] = hands[:-1]

        hand_bbox_list = [hand_bboxes]
        return hand_bbox_list

    def simple_dist(self,bbox,wrist,hand_bbox_list,bin):
        c_x = (bbox[0]+bbox[2])/2
        c_y = (bbox[1]+bbox[3])/2
        num_bbox = len(hand_bbox_list[0])
        dist_dist = np.ones((num_bbox,)) * float('inf')

        for wt in wrist:
            hands = []
            has_left = np.sum(wt[2] == -1) == 0
            has_right = np.sum(wt[5] == -1) == 0
            if not (has_left or has_right):
                continue
            if has_left:
                left_wrist_index = wt[2]
                x3, y3 = left_wrist_index[:-1]
                hands.append([x3, y3, True])

        # right hand
            if has_right:
                right_wrist_index = wt[5]
                x3, y3 = right_wrist_index[:-1]
                hands.append([x3, y3, False])
            for idx,hd in enumerate(hands):
                center_dist = np.linalg.norm(np.array([c_x, c_y]) - np.array([hd[0], hd[1]]))
                dist_dist[idx] = center_dist
            rest_hand_idx = np.argmin(dist_dist)


            if hands[rest_hand_idx][-1]:
                hand_bbox_list[0]['left_hand'] = bbox[:-1]
                new_bin = list(set(bin))
                new_bin.append(bbox[-1])
                tmp = {bbox[-1] : 'left_hand',
                       list(set(bin))[0] : 'right_hand'}
            else:
                hand_bbox_list[0]['right_hand'] = bbox[:-1]
                new_bin = list(set(bin))
                new_bin.append(bbox[-1])
                tmp = {list(set(bin))[0] : 'left_hand',
                       bbox[-1] : 'right_hand'}
        return hand_bbox_list, new_bin, tmp

    def classify_which_hand(self, raw_bbox, predict_bbox):
        hand_bbox_list = [None, ]
        num_bbox = np.array(raw_bbox).shape[0]
        hand_bboxes = dict(
            left_hand=None,
            right_hand=None
        )
        for idx, pre in enumerate(predict_bbox):

            dist_iou, dist_dist = np.ones((num_bbox,)) * float('inf'), np.ones((num_bbox,)) * float('inf')
            pre_c_x = pre[0] + pre[2] / 2
            pre_c_y = pre[1] + pre[3] / 2

            for i, raw in enumerate(raw_bbox):
                raw_c_x = (raw[0] + raw[2]) / 2
                raw_c_y = (raw[1] + raw[3]) / 2
                x1, y1, w1, h1 = raw[0], raw[1], raw[2] - raw[0], raw[3] - raw[1]

                ## comparison distance
                center_dist = np.linalg.norm(np.array([raw_c_x, raw_c_y]) - np.array([pre_c_x, pre_c_y]))
                bb_iou = dataset.bbox_IoU([x1, y1, w1, h1], pre[:-1])
                dist_iou[i] = bb_iou
                dist_dist[i] = center_dist
            raw_dist_id = np.argmin(dist_dist)
            raw_iou_id = np.argmax(dist_iou)

            if raw_dist_id == raw_iou_id:
                if pre[-1]:
                    hand_bboxes['left_hand'] = raw_bbox[raw_dist_id].copy()
                else:
                    hand_bboxes['right_hand'] = raw_bbox[raw_dist_id].copy()
            else:
                if pre[-1]:
                    hand_bboxes['left_hand'] = raw_bbox[raw_dist_id].copy()
                else:
                    hand_bboxes['right_hand'] = raw_bbox[raw_dist_id].copy()

            hand_bbox_list = [hand_bboxes]

        return hand_bbox_list

    def detect_hand_bbox_openpose(self, img, body_pose_list):
        '''
        return value: [[x, y, w, True if left hand else False]].
        width=height since the network require squared input.
        x, y is the coordinate of top left
        '''
        ratioWristElbow = 0.33
        detect_result = []
        image_height, image_width = img.shape[0:2]
        for idx, body_pose in enumerate(body_pose_list):
            hands = []
            has_left = np.sum(body_pose[[0, 1, 2]] == -1) == 0
            has_right = np.sum(body_pose[[3, 4, 5]] == -1) == 0
            if not (has_left or has_right):
                continue
            if has_left:
                left_shoulder_index, left_elbow_index, left_wrist_index = body_pose[[0, 1, 2]]
                x1, y1 = left_shoulder_index[:-1]
                x2, y2 = left_elbow_index[:-1]
                x3, y3 = left_wrist_index[:-1]
                hands.append([x1, y1, x2, y2, x3, y3, True])

            # right hand
            if has_right:
                right_shoulder_index, right_elbow_index, right_wrist_index = body_pose[[3, 4, 5]]
                x1, y1 = right_shoulder_index[:-1]
                x2, y2 = right_elbow_index[:-1]
                x3, y3 = right_wrist_index[:-1]
                hands.append([x1, y1, x2, y2, x3, y3, False])
            for x1, y1, x2, y2, x3, y3, is_left in hands:

                x = x3 + ratioWristElbow * (x3 - x2)
                y = y3 + ratioWristElbow * (y3 - y2)
                distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
                distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                width = 0.7 * max(0.6 * distanceWristElbow, 0.5 * distanceElbowShoulder)
                # x-y refers to the center --> offset to topLeft point
                # handRectangle.x -= handRectangle.width / 2.f;
                # handRectangle.y -= handRectangle.height / 2.f;
                x -= width / 2
                y -= width / 2  # width = height
                # overflow the image
                if x < 0: x = 0
                if y < 0: y = 0
                width1 = width
                width2 = width
                if x + width > image_width: width1 = image_width - x
                if y + width > image_height: width2 = image_height - y
                width = min(width1, width2)
                # the max hand box value is 20 pixels
                if width >= 20:
                    detect_result.append([int(x), int(y), int(width), int(width), is_left])

        return detect_result

class Handtrack(object):
    def __init__(self,model_path,device):
        self.model = Hand_track_Detector(model_path,device)

    def detect_hand_bbox(self,img_bgr,index):
        output = self.model.detect_hand_bbox(img_bgr,index)
        for hand_bbox in output:
            if hand_bbox is not None:
                for hand_type in hand_bbox:
                    bbox = hand_bbox[hand_type]
                    if bbox is not None:
                        x0, y0, x1, y1 = bbox
                        hand_bbox[hand_type] = np.array([x0, y0, x1 - x0, y1 - y0])

        return output

    # def detect_hand_bbox(self, img_bgr,index):
    #     """
    #     args:
    #         img_bgr: Raw image with BGR order (cv2 default). Currently assumes BGR
    #     output:
    #         body_pose_list: body poses
    #         bbox_bbox_list: list of bboxes. Each bbox has XHWH form (min_x, min_y, width, height)
    #         hand_bbox_list: each element is
    #         dict(
    #             left_hand = None / [min_x, min_y, width, height]
    #             right_hand = None / [min_x, min_y, width, height]
    #         )
    #         raw_hand_bboxes: list of raw hand detection, each element is [min_x, min_y, width, height]
    #     """
    #     output = self.model.detect_hand_bbox(img_bgr,index)
    #     # hand_bbox_list, raw_hand_bboxes = output
    #     # output_openpose = self.model.detect_hand_bbox_openpose(img_bgr)
    #
    #     # # convert raw_hand_bboxes from (x0, y0, x1, y1) to (x0, y0, w, h)
    #     # if raw_hand_bboxes is not None:
    #     #     for i in range(raw_hand_bboxes.shape[0]):
    #     #         bbox = raw_hand_bboxes[i]
    #     #         x0, y0, x1, y1 = bbox
    #     #         raw_hand_bboxes[i] = np.array([x0, y0, x1 - x0, y1 - y0])
    #
    #     # convert hand_bbox_list from (x0, y0, x1, y1) to (x0, y0, w, h)
    #     for hand_bbox in output:
    #         if hand_bbox is not None:
    #             for hand_type in hand_bbox:
    #                 bbox = hand_bbox[hand_type]
    #                 if bbox is not None:
    #                     x0, y0, x1, y1 = bbox
    #                     hand_bbox[hand_type] = np.array([x0, y0, x1 - x0, y1 - y0])
    #
    #     return output
