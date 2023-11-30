import cv2

# 设置分辨率  
resolution = (1920, 1080)  
  
# 打开相机设备，设置分辨率  
camera = cv2.VideoCapture(0)  # 使用默认的USB相机设备，如果有多个设备，可以改变参数为相应的设备ID  
camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  
  
# 检查是否成功打开相机  
if not camera.isOpened():  
    print("无法打开相机")  
    exit()  
  
# 循环读取相机数据并显示图像  
while True:  
    # 从相机读取图像数据  
    ret, frame = camera.read()  
      
    # 检查是否成功读取图像  
    if not ret:  
        print("无法读取图像")  
        break  
      
    # 显示图像  
    cv2.imshow("Camera", frame)  
      
    # 等待按下 'q' 键退出或 1ms后自动显示下一帧图像  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
    if cv2.waitKey(1) & 0xFF == ord('s'):  
        cv2.imwrite("1.jpg", frame)
        print("1.jpg")
  
# 释放相机设备和关闭窗口  
camera.release()  
cv2.destroyAllWindows()