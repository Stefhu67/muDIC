# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
sys.path.extend([abspath(".")])
import muDIC as dic
import logging
import matplotlib.pyplot as plt
from muDIC.stereo.calibration import Calibration, Distortion
import glob

# Set the amount of info printed to terminal during analysis
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

path_l = glob.glob(r'./Image_test_L/*')
path_r = glob.glob(r'./Image_test_R/*')

calib_path_l = glob.glob(r'./Image_calib_L/*')
calib_path_r = glob.glob(r'./Image_calib_R/*')
Calibration.crop(imageStack=path_l, calib_path=calib_path_l, side='L')
# We assume that the calibration required here is the same for the 2 cameras (if they are the same), so calib_path=calib_path_l again
Calibration.crop(imageStack=path_r, calib_path=calib_path_l, side='R')

# We are going to work with the path which containing all the cropted images 
cropted_path=glob.glob(r'./Cropted_images/*')
Distortion.comparison(imageStack=cropted_path)

# Allow the user to recover the cropted images of each cameras of both sides
path_l_cropted=glob.glob(r"./Cropted_images/*L.tif")
path_r_cropted=glob.glob(r"./Cropted_images/*R.tif")

