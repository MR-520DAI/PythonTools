import os
import math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def loader_func(path):
    img = Image.open(path)
    return img

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius -
                               left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

class LaneClsDataset(torch.utils.data.Dataset):

    def __init__(self, path, list_path, img_transform=None):
        super(LaneClsDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()

    def __getitem__(self, index):
        ratio_x = 8 * 2.05
        ratio_y = 8 * 1.84375
        name = self.list[index].split()[0]
        img_path = "/workspace/Ultra-Fast-Lane-Detection/data/CULane" + name
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        line_index = [int(self.list[index].split()[-4]), int(self.list[index].split()[-3]),
                      int(self.list[index].split()[-2]), int(self.list[index].split()[-1])]

        label_path = "/workspace/Ultra-Fast-Lane-Detection/data/CULane" + name[:-3] + "lines.txt"
        
        label_hm = np.zeros((1, 20, 50))
        label_row = np.zeros((4, 40))
        label_range = np.zeros((4, 40))

        loc = 0

        with open(label_path, "r") as f:
            data = f.readlines()
            if line_index[1] == 1 and line_index[2] == 1:
                x1_list = []
                y1_list = []
                x2_list = []
                y2_list = []
                if line_index[0] == 1:
                    line1 = data[1]
                    line2 = data[2]
                else:
                    line1 = data[0]
                    line2 = data[1]
                line1 = line1.strip("\n").strip(" ").split(" ")
                line2 = line2.strip("\n").strip(" ").split(" ")
                for i in range(0, len(line1), 2):
                    x1_list.append(float(line1[i]))
                    y1_list.append(float(line1[i+1]))
                for i in range(0, len(line2), 2):
                    x2_list.append(float(line2[i]))
                    y2_list.append(float(line2[i+1]))
                x1_list = np.array(x1_list)
                y1_list = np.array(y1_list)
                x2_list = np.array(x2_list)
                y2_list = np.array(y2_list)
                z1 = np.polyfit(x1_list, y1_list, 1)
                z2 = np.polyfit(x2_list, y2_list, 1)
                vx = (z1[1]-z2[1]) / (z2[0]-z1[0])
                vy = z1[0] * vx + z1[1]
                center = (round(vx / ratio_x / 2), round(vy / ratio_y / 2))
                draw_umich_gaussian(label_hm[0], center, 3)

            for line in data:
                while line_index[loc] == 0:
                    loc+=1
                line_data = []
                line = line.strip("\n").strip(" ").split(" ")
                lenth = len(line)
                for i in range(0, lenth, 2):
                    x = float(line[i]) / ratio_x
                    y = float(line[i+1]) / ratio_y
                    line_data.append((x, y))

                y_max = min(line_data[0][1], 39.)
                y_min = max(0., line_data[-1][1])
                #print(line_data)
                new_line = []
                for y in range(math.ceil(y_min), int(y_max+1)):
                    for i in range(len(line_data)-1):
                        x1 = line_data[i][0]
                        y1 = line_data[i][1]
                        x2 = line_data[i+1][0]
                        y2 = line_data[i+1][1]
                        if y < y1 and y > y2:
                            if x1 != x2:
                                A = (y2 - y1) / (x2 - x1)
                                B = y2 - A * x2
                                x = (y - B) / A
                                if x < 0 or x > 100:
                                    break
                                new_line.append((x, y))
                            else:
                                x = x1
                                if x < 0 or x > 100:
                                    break
                                new_line.append((x, y))
                            break
            
                for xy in new_line:
                    x = xy[0]
                    y = int(xy[1])
                    label_row[loc][y] = x
                    label_range[loc][y] = 1

                loc+=1

        mask = label_range
        return img, label_row, label_range, label_hm, mask

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    img_transform = transforms.Compose([
        transforms.Resize((320, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset = LaneClsDataset('/workspace/Ultra-Fast-Lane-Detection/data/CULane', 
    '/workspace/Ultra-Fast-Lane-Detection/data/CULane/list/train_gt.txt', img_transform)

    sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=0)
    for i, data in enumerate(train_loader):
        img, label_row, label_range, mask, label_hm = data
        print(label_row.shape)