import numpy as np
import cv2 
import glob


class Calibration(object):
    def __init__(self):
        pass
    
    def cameras_calib(path, points_in_width=6, points_in_height=9):
        """
        Aims to calibrate the cameras so that we can remove distortion afterwards
        ---------
        path:
            The folder in which we can find the differnets pictures of the chessboard. 
            It is important that the photos are taken from the same camera and from diferents angles. 
        width:
            Number of points required in width for calibration
        height:
            Number of points required in height for calibration
        """
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points 
        objp = np.zeros((points_in_width*points_in_height,3), np.float32)
        objp[:,:2] = np.mgrid[0:points_in_height,0:points_in_width].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for fname in path:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (points_in_height,points_in_width),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (points_in_height,points_in_width), corners2,ret)
                #If you want to see the images of the calibration (the chessboard) uncomment the 3 lines below
                #cv2.imshow('image',img)
                #cv2.waitKey(500) #It's the display time 

        #cv2.destroyAllWindows()
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        return([ret, mtx, dist, rvecs, tvecs])

    def crop(imageStack,calib_path, side='R'):
        """
        Allow the reshape the picture and to avoid curves on the sides of images 
        --------
        cost: 'R' or 'L'
            The cost which the picture was taken 
        """
        nb_frame=1
        ret=Calibration.cameras_calib(calib_path,6,9)[0]
        mtx=Calibration.cameras_calib(calib_path,6,9)[1]
        dist=Calibration.cameras_calib(calib_path,6,9)[2]
        rvecs=Calibration.cameras_calib(calib_path,6,9)[3]
        tvecs=Calibration.cameras_calib(calib_path,6,9)[4]
        for frame in imageStack:
            img = cv2.imread(frame)
            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
            cv2.imwrite(r'./Cropted_images/Image'+str(nb_frame)+side+'.png',dst)            
            nb_frame+=1

class Distortion(object):
    def __init__(self):
        pass

    def comparison(imageStack):
        from matplotlib import pyplot as plt 
    
        # read left and right images 
        for i in range (1,int(len(imageStack)/2)):
            imgR = cv2.imread(r'./Cropted_images/Image'+str(i)+'R.png', cv2.IMREAD_GRAYSCALE) 
            imgL = cv2.imread(r'./Cropted_images/Image'+str(i)+'L.png', cv2.IMREAD_GRAYSCALE) 
            # creates StereoBm object, it is possible to play with the value of numDisparities but it must remain a multiple of 16
            stereo = cv2.StereoBM_create(numDisparities = 16*2, blockSize = 15) 

            # computes disparity 
            disparity = stereo.compute(imgL, imgR) 
            
            # displays image as grayscale and plotted 
            plt.imshow(disparity, 'gray') 
            plt.show()


