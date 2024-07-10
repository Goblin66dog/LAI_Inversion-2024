import glob
import os.path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from sklearn.metrics import mean_squared_error, r2_score

import DataReader


def ReadByBands(image_item, band):
    image_item = gdal.Open(image_item.GetSubDatasets()[band][0], gdal.GA_ReadOnly)
    image_array = np.array(image_item.ReadAsArray())
    image_array[image_array < 0 ] = 0
    return image_array


def Data_List(pack_path):
    data_list = glob.glob(os.path.join(pack_path,"*.hdf"))
    return data_list

def Cal_Ref2NDVI(b1, b2):
    ndvi=(b2-b1)/(b2+b1)
    ndvi[np.isnan(ndvi)]=np.nanmin(ndvi)
    return ndvi
def scat_fig(y_true, y_pred):
    title = "0"
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # 计算决定系数（R²）
    r_squared = r2_score(y_true, y_pred)
    # 计算相对均方根误差（RMRSE）
    rmrse = rmse / (np.mean(y_true) - np.min(y_true)) if np.mean(y_true) - np.min(y_true) != 0 else np.nan
    # 计算平均绝对误差（MAE）
    mae = np.mean(np.abs(y_true - y_pred))
    # 计算平均绝对百分比误差（MAPE）
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.plot((0, 1), (0, 1), linewidth=1, transform=ax.transAxes, ls='--', c='k', label="1:1 line", alpha=0.5)
    ax.plot(y_true, y_pred, 'o', c='blue', markersize=2)
    bbox = dict(boxstyle="round", fc='1', alpha=0.)
    bbox = bbox
    plt.text(0.05, 0.65, "$R^2: %.2f$\n$RMSE: %.2f$\n$RMRSE: %.2f$\n$MAE: %.2f$\n$MAPE: %.2f$" % (
    (r_squared), (rmse), (rmrse), mae, mape),
             transform=ax.transAxes, size=7, bbox=bbox, fontdict={'family': 'Times New Roman', 'weight': 'bold'})
    # plt.text(9, 0.2, "(%s)" % (title))
    ax.set_xlabel('Ground Truth', fontsize=7, fontdict={'family': 'Arial', 'size': 14, 'weight': 'bold'})
    ax.set_ylabel("Calculated", fontsize=7, fontdict={'family': 'Arial', 'size': 14, 'weight': 'bold'})
    ax.set(xlim=(0, 1), ylim=(0, 1))
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(r"C:\Users\Vitch\Desktop\Quantitative RS\Figure_1.png", dpi=400)

if __name__=="__main__":
    pack_path = r"C:\Users\Vitch\Desktop\Quantitative RS"
    data_list = Data_List(pack_path)
    reflection_path = data_list[0]
    ndvi_path       = data_list[1]
    lai_path        = data_list[2]

# Reflection->NDVI
    reflection_item  = gdal.Open(reflection_path)
    ndvi_item        = gdal.Open(ndvi_path)
    lai_item         = gdal.Open(lai_path)
    for i in lai_item.GetSubDatasets():
        print(i)

    ref_b1 = ReadByBands(reflection_item, 0)
    ref_b2 = ReadByBands(reflection_item, 1)

    cal_ndvi = cv2.resize(Cal_Ref2NDVI(ref_b1, ref_b2), None, None, 0.5, 0.5, cv2.INTER_NEAREST)
# save tif
#     driver = gdal.GetDriverByName('GTiff')
#     dataset = driver.Create("Cal_NDVI.TIF",
#                             cal_ndvi.shape[0],
#                             cal_ndvi.shape[1],
#                             1,
#                             gdal.GDT_Float32)
#     dataset.SetGeoTransform(ndvi_item.GetGeoTransform())  # 写入仿射变换参数
#     dataset.SetProjection(ndvi_item.GetProjection())  # 写入投影
#     dataset.GetRasterBand(1).WriteArray(cal_ndvi)
#     dataset.FlushCache()  # 确保所有写入操作都已完成
#     dataset = None
# 散点图
    ndvi = ReadByBands(ndvi_item, 0)
    ndvi = cv2.normalize(ndvi, None, 0, 1,cv2.NORM_MINMAX,cv2.CV_32F)
    plt.subplot(2,1,1)
    cal_ndvi[cal_ndvi<=0]=0
    cal = plt.imshow(cal_ndvi,"coolwarm")
    # plt.colorbar(cal)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(r"C:\Users\Vitch\Desktop\Quantitative RS\cal_ndvi.png",dpi=400)
    plt.subplot(2,2,1)
    nd = plt.imshow(ndvi,"coolwarm")
    # plt.colorbar(nd)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # plt.savefig(r"C:\Users\Vitch\Desktop\Quantitative RS\ndvi.png",dpi=400)
    scat_fig(cal_ndvi[500:600,675:700],ndvi[500:600,675:700])



