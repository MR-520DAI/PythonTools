import cv2
import numpy as np
import torch
from PIL import Image
from LaneDetModel import LaneDect
import torchvision.transforms as transforms

def compute_locations(shape):
    pos = torch.arange(
        0, shape[-1], step=1, dtype=torch.float32)
    pos = pos.reshape((1, 1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], shape[2], 1)
    return pos

device = torch.device('cuda:0')

net = LaneDect()
state_dict = torch.load("./model/LaneDect_29.pth", map_location='cpu')
net.load_state_dict(state_dict)
net.to(device)
net.eval()

img_transform = transforms.Compose([
        transforms.Resize((320, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

root = "/workspace/Ultra-Fast-Lane-Detection/data/CULane"
save_root = "/workspace/LaneDect/txt_result"
with open("/workspace/Ultra-Fast-Lane-Detection/data/CULane/list/test.txt", "r") as f:
    data = f.readlines()
    for i in data:
        img_path = i.strip("\n")

        img = Image.open(root + img_path)
        img = img_transform(img).unsqueeze(0)

        seg, ranges, hm = net(img.to(device))

        seg_softmax = torch.softmax(seg, dim=-1)
        pos = compute_locations(seg_softmax.size())
        row_pos = torch.sum(pos.to(device) * seg_softmax, dim=3) + 0.5
        row_pos = row_pos.squeeze()

        ranges_softmax = torch.softmax(ranges, dim=1)
        col_pos = torch.argmax(ranges_softmax, dim=1)
        col_pos = col_pos.squeeze()
        ranges_softmax_cpu = ranges_softmax.cpu()
        numpy_range = ranges_softmax_cpu.detach().numpy().squeeze()

        txt_path = save_root + img_path[:-3] + "lines.txt"
        print(txt_path)
        with open(txt_path, "w") as f:
            for i in range(4):
                if torch.sum(col_pos[i]) == 0:
                    continue
                if torch.sum(col_pos[i]) < 5:
                    continue
                for j in range(40):
                    if numpy_range[1][i][39-j] > numpy_range[0][i][39-j] and numpy_range[1][i][39-j] > 0.5:
                        y = (39-j) * 8 * 1.84375
                        x = float(row_pos[i][39-j]) * 8 * 2.05
                        x = round(x, 2)
                        f.write(str(x) + " " + str(int(y)) + " ")
                f.write("\n")