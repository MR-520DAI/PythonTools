import os
import math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def loader_func(path):
    img = Image.open(path)
    return img

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
        
        loc = 0
        lane_list = np.zeros((4, 41))

        with open(label_path, "r") as f:
            data = f.readlines()

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
                                if x <= 0 or x >= 100:
                                    continue
                                new_line.append((x, y))
                            else:
                                x = x1
                                if x <= 0 or x >= 100:
                                    continue
                                new_line.append((x, y))
                            break
                
                for xy in new_line:
                    x = xy[0] * 8
                    y = int(xy[1])
                    lane_list[loc][y] = x

                lane_list[loc][0] = 1
                loc+=1
        return img, lane_list

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
        img, lane_list = data
        print(lane_list.shape)