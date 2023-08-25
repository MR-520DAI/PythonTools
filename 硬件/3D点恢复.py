import cv2
import numpy as np

def XYZID52(L):
    print("ID2")

    Xtheta = -0.96445
    Ytheta = 0.012026
    Ztheta = 0.616356
    Xd = -75.3648
    Yd = -2.43206
    Zd = -36.2765
    
    print("X1,Y1,Z1: ", L*np.sin(Xtheta) + Xd, ",", L*np.sin(Ytheta) + Yd, ",", L*np.sin(Ztheta) + Zd)

    Xtheta = -0.941817
    Ytheta = 0.0148392
    Ztheta = 0.607196
    Xd = -10.406
    Yd = 1.04591
    Zd = 34.1819
    
    print("X2,Y2,Z2: ", L*np.sin(Xtheta) + Xd, ",", L*np.sin(Ytheta) + Yd, ",", L*np.sin(Ztheta) + Zd)

def XYZID42(L):
    print("ID2")

    Xtheta = 0.890688
    Ytheta = 0.024472
    Ztheta = 0.671737
    Xd = 68.5806
    Yd = -0.68953
    Zd = -37.0284
    
    print("X1,Y1,Z1: ", L*np.sin(Xtheta) + Xd, ",", L*np.sin(Ytheta) + Yd, ",", L*np.sin(Ztheta) + Zd)

    Xtheta = 0.87496
    Ytheta = 0.00982316
    Ztheta = 0.687048
    Xd = 9.15658
    Yd = -0.938295
    Zd = 40.6318
    
    print("X2,Y2,Z2: ", L*np.sin(Xtheta) + Xd, ",", L*np.sin(Ytheta) + Yd, ",", L*np.sin(Ztheta) + Zd)

def XYZID22(L):
    print("ID2")

    Xtheta = 0.013166
    Ytheta = -1.05442
    Ztheta = 0.515399
    Xd = -4.31094
    Yd = -67.2622
    Zd = -39.6094
    
    print("X1,Y1,Z1: ", L*np.sin(Xtheta) + Xd, ",", L*np.sin(Ytheta) + Yd, ",", L*np.sin(Ztheta) + Zd)

    Xtheta = -0.0105752
    Ytheta = -1.08501
    Ztheta = 0.488143
    Xd = -1.152
    Yd = -1.64148
    Zd = 33.9198
    
    print("X2,Y2,Z2: ", L*np.sin(Xtheta) + Xd, ",", L*np.sin(Ytheta) + Yd, ",", L*np.sin(Ztheta) + Zd)

def XYZID32(L):
    print("ID2")

    Xtheta = -0.01239
    Ytheta = 1.1158
    Ztheta = 0.453369
    Xd = 3.39252
    Yd = 74.6961
    Zd = -48.112
    
    print("X1,Y1,Z1: ", L*np.sin(Xtheta) + Xd, ",", L*np.sin(Ytheta) + Yd, ",", L*np.sin(Ztheta) + Zd)

    Xtheta = 0.00475623
    Ytheta = 1.09403
    Ztheta = 0.477748
    Xd = 1.63393
    Yd = 3.15434
    Zd = 37.5912
    
    print("X2,Y2,Z2: ", L*np.sin(Xtheta) + Xd, ",", L*np.sin(Ytheta) + Yd, ",", L*np.sin(Ztheta) + Zd)

if __name__ == "__main__":
    L2 = [1518, 1421, 1349, 1200, 1069, 2000]
    
    for i in L2:
        XYZID32(i)