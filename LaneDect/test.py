from re import T
import torch
import torch.nn as nn

a = torch.tensor([[[1.,2,3],[4,5,6],[7,8,9]],
[[10,11,12],[13,14,15],[16,17,18]],
[[19,20,21],[22,23,24],[25,26,27]]])
a = a.reshape((1,3,3,3))

b = torch.tensor([[[1.,1,1],[1,1,1],[1,1,1]],
[[1,1,1],[1,1,1],[1,1,1]],
[[1,1,1],[1,1,1],[1,1,1]]])
b = b.reshape(1,3,3,3)

c = a.new_zeros([1, 3, 3, 3])

for i in range(3):
    if i > 0:
        c[:, i, :, i:] = torch.norm(a[:, :, :, i:] - b[:, :, :, :3-i], 1, 1, keepdim=False)
    else:
        c[:, i, :, :] = torch.norm(a[:, :, :, :] - b[:, :, :, :], 1, 1, keepdim=False)

print(c)
