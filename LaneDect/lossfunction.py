from operator import index, le
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def MatchLane(pre_point, label, H):
    match_index = []
    for i in range(H):
        if label[i][0] == 0:
            continue
        iou_mat = pre_point - label[i][1:]
        iou_mat = torch.abs(iou_mat)
        iou_mat = torch.sum(iou_mat, dim=1)
        min_index = torch.argmin(iou_mat)
        if min_index.item() not in match_index:
            match_index.append(min_index.item())
        else:
            min_index = random.randint(0,31)
            while min_index in match_index:
                min_index = random.randint(0,31)
            min_iou = iou_mat[min_index]
            for j in range(32):
                if j in match_index:
                    continue
                if iou_mat[j] < min_iou:
                    min_iou = iou_mat[j]
                    min_index = j
            match_index.append(min_index)
    return match_index


class LaneLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(LaneLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.nll = nn.NLLLoss(ignore_index=255)

    def forward(self, cls_exit, point, label):
        N, H, W = label.shape
        pre_point = point.clone()
        pre_point *= 799

        lane_num = 0
        point_loss = 0
        cls_exit_loss = 0

        for i in range(N):
            match_index = MatchLane(pre_point[i], label[i], H)
            # ID = 0
            # match_index = []
            # for j in range(4):
            #     if label[i][j][0] != 0:
            #         match_index.append(ID)
            #         ID+=1
            target_cls = point.new_zeros(point.shape[1]).long()
            target_cls[match_index] = 1
            factor = torch.pow(1. - cls_exit[i], self.gamma)
            focal = factor * torch.log(cls_exit[i])
            cls_exit_loss += self.nll(focal, target_cls)

            if len(match_index) == 0:
                continue
            else:
                loc = 0
                for j in match_index:
                    while label[i][loc][0] == 0:
                        loc+=1
                    point_loss += self.l1(pre_point[i][j], label[i][loc][1:])
                    lane_num += 1
                    loc+=1

        return cls_exit_loss / N + 0.4 * point_loss / (lane_num + 1e-6)

if __name__ == '__main__':
    x = torch.tensor([[0.1, 0., -1], [0.1, 0.1, 0.2]])
    print(x.shape)
    mask = torch.gt(x, 0)
    print(mask)