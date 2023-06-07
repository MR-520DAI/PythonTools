import os
import shutil

sourceroot = "E:\\c_project\\learn\\Open3D-master\\examples\\python\\reconstruction_system\\dataset\\aobi_rgbd\\maoge\\"
targetroot = "E:\\c_project\\tools\\COLMAP-3.8\\project\\"

colorName = os.listdir(sourceroot + "color")
depthName = os.listdir(sourceroot + "depth")

iD = 0
for i in range(0, len(colorName), 20):
    oldColorName = sourceroot + "color\\" + colorName[i]
    newColorName = targetroot + "images\\" + str(iD) + ".jpg"
    shutil.copy(oldColorName, newColorName)

    oldDepthName = sourceroot + "depth\\" + depthName[i]
    newDepehName = targetroot + "depth\\" + str(iD) + ".png"
    shutil.copy(oldDepthName, newDepehName)
    iD += 1
    print(iD)