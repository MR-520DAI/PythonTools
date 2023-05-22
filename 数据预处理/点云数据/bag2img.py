import cv2
import rospy
import rosbag
from cv_bridge import CvBridge
import sensor_msgs

topic_depth = "/camera/depth/image_raw"
topic_color = "/camera/color/image_raw"

rospy.init_node("bag_to_images", anonymous=True)


bridge = CvBridge()

bag = rosbag.Bag("./test/5.097.bag")

num = 0
for topic, msg, t in bag.read_messages(topics=topic_depth):
    img = bridge.imgmsg_to_cv2(msg, "16UC1")
    
    cv2.imwrite("./test/depth/" + "%06d"%num + ".png", img)
    print("depth", num)
    num+=1

num = 0
for topic, msg, t in bag.read_messages(topics=topic_color):
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    cv2.imwrite("./test/color/"+"%06d"%num+".jpg", img)
    print("color", num)
    num+=1

