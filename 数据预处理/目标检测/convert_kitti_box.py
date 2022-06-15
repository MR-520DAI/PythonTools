import os
import cv2

# cv2:(h, w, c)

nc = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

root = "Y:\\data\\kitti\\training\\"

file_name = os.listdir(root + "image_2")

for img_name in file_name:
    print(img_name)
    img = cv2.imread(root + "image_2\\" + img_name)
    h = img.shape[0]
    w = img.shape[1]
    with open(root + "label_2\\" + img_name.split(".")[0] + ".txt", "r") as f:
        txt_data = f.readlines()
        for data in txt_data:
            ID = nc.index(data.split(" ")[0])
            x = ((float(data.split(" ")[6]) + float(data.split(" ")[4])) / 2.0 - 1) / w
            y = ((float(data.split(" ")[7]) + float(data.split(" ")[5])) / 2.0 - 1) / h
            width = (float(data.split(" ")[6]) - float(data.split(" ")[4])) / w
            height = (float(data.split(" ")[7]) - float(data.split(" ")[5])) / h
            with open("labels\\" + img_name.split(".")[0] + ".txt", "a+") as nf:
                nf.write(str(ID) + " " + str(x)[:8] + " " + str(y)[:8] + " " + str(width)[:8] + " " + str(height)[:8] + "\n")
