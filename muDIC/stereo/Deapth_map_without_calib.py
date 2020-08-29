# import OpenCV and pyplot  
from matplotlib import pyplot as plt 
import cv2
import glob
from PIL import Image

Images_R = glob.glob(r'./Image_test_R/*')
Images_L = glob.glob(r'./Image_test_L/*')

for i in range (len(Images_R)):
    # read left and right images 

    imgR = cv2.imread(Images_R[i], cv2.IMREAD_GRAYSCALE) 
    imgL = cv2.imread(Images_L[i], cv2.IMREAD_GRAYSCALE) 
    # creates StereoBm object  
    stereo = cv2.StereoBM_create(numDisparities = 16*13, blockSize = 9) 

    # computes disparity 
    disparity = stereo.compute(imgL, imgR) 

    # displays image as grayscale and plotted 
    plt.imshow(disparity, vmin=0, vmax=1200, cmap='viridis')
    plt.show()