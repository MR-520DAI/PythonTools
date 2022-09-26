import tqdm
import torch
from LaneDetModel import LaneDect
from lossfunction import LaneLoss
from dataset import LaneClsDataset
import torchvision.transforms as transforms

if __name__ == '__main__':
    img_transform = transforms.Compose([
        transforms.Resize((320, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    device = torch.device('cuda:0')

    train_dataset = LaneClsDataset('/workspace/Ultra-Fast-Lane-Detection/data/CULane', 
    '/workspace/Ultra-Fast-Lane-Detection/data/CULane/list/train_gt.txt', img_transform)
    sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=0)
    train_loader = tqdm.tqdm(train_loader)

    net = LaneDect()
    # state_dict = torch.load("/workspace/LaneDect/model/LaneDect_2.pth")
    # net.load_state_dict(state_dict)
    # torch.save(net.state_dict(), "/workspace/LaneDect/model/LaneDect_" + ".pth")
    net.to(device)

    loss = LaneLoss()

    training_params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.Adam(training_params, lr=0.00003, weight_decay=0.0001)

    for epoch in range(0, 50):
        for i, data in enumerate(train_loader):
            net.train()
            img, label_seg, label_offset, mask_offset = data
            pre_seg, pre_offset = net(img.to(device))

            ls = loss(pre_seg, pre_offset, label_seg.to(device).long(),
            label_offset.to(device).float(), mask_offset.to(device).float())
            optimizer.zero_grad()
            print(ls)
            ls.backward()
            optimizer.step()
            train_loader.set_postfix(loss='%.3f'%float(ls))

        torch.save(net.state_dict(), "/workspace/LaneDect/model/LaneDect_" + str(epoch) + ".pth")
