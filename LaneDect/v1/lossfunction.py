import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def compute_locations(shape, device):
    pos = torch.arange(
        0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], shape[2], 1)
    return pos

class LaneLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(LaneLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.l1 = nn.L1Loss(reduction='sum')
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, inputs_seg, inputs_range, inputs_hm, targets_pos, 
    targets_range, targets_hm, mask):
        # 消失点损失
        N, C, H, W = inputs_seg.shape
        gt = targets_hm
        pred = inputs_hm
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        vp_loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                                2) * neg_weights * neg_inds
        if torch.sum(pos_inds) != 0:
            pos_loss = pos_loss.sum() / N
            neg_loss = neg_loss.sum() / N
            vp_loss = vp_loss - (pos_loss + neg_loss)
        else:
            vp_loss = torch.tensor(0, dtype=torch.float32).to(pred.device)
        
        # 横坐标损失
        seg_softmax = F.softmax(inputs_seg, dim=-1)
        pos = compute_locations(seg_softmax.size(), device=seg_softmax.device)
        row_pos = torch.sum(pos * seg_softmax, dim=3) + 0.5
        row_loss = self.l1(row_pos * mask, targets_pos * mask)
        row_loss = torch.sum(row_loss) / (torch.sum(mask).float() + 1e-4)

        # 纵坐标损失
        range_softmax = F.softmax(inputs_range, dim=1)
        range_loss = self.CrossEntropyLoss(range_softmax, targets_range)
        
        return vp_loss + row_loss + range_loss



if __name__ == '__main__':
    x = torch.tensor([[0.1, 0.1, 0.2], [0.1, 0.1, 0.2]])
    x = x.view([1, 2, 1, 3])

    y = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
    y = y.view([1,2,1,3])

    loss = nn.BCELoss(reduce=False)
    all_loss = loss(x, y)
    # print(all_loss)

    pos_num = torch.sum(torch.gt(y, 0))
    neg_num = torch.sum(torch.lt(y, 1))
    pos_weight = torch.gt(y, 0).float()
    neg_weight = torch.lt(y, 1).float()

    # print(pos_weight)
    print(pos_num.sum().float())
    pos_loss = all_loss * pos_weight
    neg_loss = all_loss * neg_weight
    print(pos_loss.sum() / 2 + neg_loss.sum() / 4)
    print(all_loss.sum() / 6)
