import glob
import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from sklearn.metrics import mean_squared_error, r2_score

from DataReader import Dataset

def Data_List(pack_path):
    data_list = glob.glob(os.path.join(pack_path,"*.tif"))
    return data_list

def Cal_Ref2NDVI(b1, b2):
    return (b2-b1)/(b2+b1+1e-10)

if __name__ == "__main__":
    pack_path = r"C:\Users\Vitch\Desktop\定量遥感导论"
    data_list = Data_List(pack_path)
    reflection_path = data_list[0]
    ndvi_path       = data_list[1]
    # lai_path        = data_list[2]

    reflection = Dataset(reflection_path)
    reflection_array = reflection.array[reflection.array<0]=1e-8
    cal_ndvi = Cal_Ref2NDVI(reflection.array[0], reflection.array[1])

    plt.subplot(2,1,1)
    plt.imshow(cal_ndvi)
    # plt.subplot(2,1,2)
    # plt.imshow(ndvi)
    plt.show()
    # scat_fig(ndvi,cal_ndvi)
