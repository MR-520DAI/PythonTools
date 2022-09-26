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
        self.nll = nn.NLLLoss(ignore_index=255)

    def forward(self, inputs_seg, inputs_offset, targets_seg, 
    targets_offset, mask_offset):
        # 分割损失
        rows = inputs_seg.permute(0, 3, 1, 2).contiguous()
        factor = torch.pow(1.-rows, self.gamma)
        log_score = torch.log(rows)
        log_score = factor * log_score
        row_loss = self.nll(log_score, targets_seg)
        # 偏移损失
        pos_num = torch.sum(mask_offset)
        neg_num = torch.sum(1-mask_offset)

        offset_pos = inputs_offset * mask_offset
        offset_neg = inputs_offset * (1-mask_offset)

        offset_loss_pos = self.l1(offset_pos, targets_offset) / (pos_num + 1e-9)
        offset_loss_neg = self.l1(offset_neg, targets_offset * (1-mask_offset)) / (neg_num + 1e-9)

        
        return offset_loss_pos + offset_loss_neg + row_loss



if __name__ == '__main__':
    x = torch.tensor([[0.1, 0., -1], [0.1, 0.1, 0.2]])
    print(x.shape)
    mask = torch.gt(x, 0)
    print(mask)