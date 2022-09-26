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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
    train_loader = tqdm.tqdm(train_loader)

    net = LaneDect()
    # state_dict = torch.load("/workspace/LaneDect/model/LaneDect_0_29999.pth")
    # net.load_state_dict(state_dict)
    # torch.save(net.state_dict(), "/workspace/LaneDect/model/LaneDect_" + ".pth")
    net.to(device)

    loss = LaneLoss()

    training_params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.Adam(training_params, lr=0.0001, weight_decay=0.0001)

    for epoch in range(0, 50):
        for i, data in enumerate(train_loader):
            net.train()
            img, label = data
            pre_exit, pre_point = net(img.to(device))

            ls = loss(pre_exit, pre_point, label.to(device))
            optimizer.zero_grad()
            print(ls)
            ls.backward()
            optimizer.step()
            train_loader.set_postfix(loss='%.3f'%float(ls))
            # if (i+1) % 100 == 0:
            #     torch.save(net.state_dict(), "/workspace/LaneDect/model/LaneDect_" + str(epoch) + "_" + str(i) + ".pth")
        torch.save(net.state_dict(), "/workspace/LaneDect/model/LaneDect_" + str(epoch) + ".pth")
