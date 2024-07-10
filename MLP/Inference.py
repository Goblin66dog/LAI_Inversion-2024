import glob
import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import DataReader
import DataLoader
from model import MLP
from osgeo import gdal

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
    plt.text(0.05, 0.8, "$R^2: %.2f$\n$RMSE: %.2f$\n$RMRSE: %.2f$\n$MAE: %.2f$"
                         # "\n$MAPE: %.2f$"
             % (
    (r_squared), (rmse), (rmrse), mae),
             transform=ax.transAxes, size=7, bbox=bbox, fontdict={'family': 'Times New Roman', 'weight': 'bold'})
    # plt.text(9, 0.2, "(%s)" % (title))
    ax.set_xlabel('Ground Truth', fontsize=7, fontdict={'family': 'Arial', 'size': 14, 'weight': 'bold'})
    ax.set_ylabel("Prediction", fontsize=7, fontdict={'family': 'Arial', 'size': 14, 'weight': 'bold'})
    ax.set(xlim=(0, 50), ylim=(0, 50))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 10, 20, 30, 40, 50])
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.savefig(r"C:\Users\Vitch\Desktop\Quantitative RS\NDVI-LAI_Precision.png",dpi=400)

def SaveWithGeoInfo(item, image):
    axs = [0, 1], [0, 1]

    image = np.transpose(image, axs[0])
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create("prediction.TIF",
                            image.shape[1],
                            image.shape[0],
                            1,
                            gdal.GDT_Float32)
    image = np.transpose(image, axs[1])
    dataset.SetGeoTransform(item.geotrans)  # 写入仿射变换参数
    dataset.SetProjection(item.proj)  # 写入投影
    dataset.GetRasterBand(1).WriteArray(image)
    dataset.FlushCache()  # 确保所有写入操作都已完成
    dataset = None

def deploy(model_path,image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = torch.tensor(image, dtype=torch.float32)
    model = MLP(in_channels=1, out_channels=1, hidden_size=512)
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载pth文件
    model.to(device=device)
    model = model.eval()

    output = []
    for rows in image:
        # for cols in rows:
        rows = rows.reshape(1200,1)
        rows = rows.to(device=device, dtype=torch.float32)
        rows = model(rows)
        rows = np.array(rows.data.cpu())
        output.append(rows)
    output = np.array(output)[:,:,0]
    # SaveWithGeoInfo(image_item, output)
    plt.figure()
    heatmap=plt.imshow(output,"coolwarm")
    plt.colorbar(heatmap)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    # plt.savefig(r"C:\Users\Vitch\Desktop\Quantitative RS\NDVI-LAI.png",dpi=400,bbox_inches='tight')
    return output

def ReadByBands(image_item, band):
    image_item = gdal.Open(image_item.GetSubDatasets()[band][0], gdal.GA_ReadOnly)
    image_array = np.array(image_item.ReadAsArray())
    image_array[image_array < 0] = 255
    # image_array = cv2.normalize(image_array, None, 0, 1,cv2.NORM_MINMAX,cv2.CV_32F)

    return image_array


def Data_List(pack_path):
    data_list = glob.glob(os.path.join(pack_path,"*.hdf"))
    return data_list

if "__main__" == __name__:
    pack_path = r"C:\Users\Vitch\Desktop\Quantitative RS"
    data_list = Data_List(pack_path)
    reflection_path = data_list[0]
    ndvi_path = data_list[1]
    lai_path = data_list[2]

    # Reflection->NDVI
    reflection_item = gdal.Open(reflection_path)
    ndvi_item = gdal.Open(ndvi_path)
    lai_item = gdal.Open(lai_path)

    ndvi = ReadByBands(ndvi_item, 0)
    lai = ReadByBands(lai_item, 1)
    lai[lai > 100] = 0
    ndvi[ndvi==255]= 0
    ndvi = cv2.normalize(ndvi, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    plt.figure("GT")
    heatmap=plt.imshow(lai,"coolwarm")
    plt.colorbar(heatmap)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(r"C:\Users\Vitch\Desktop\Quantitative RS\GT-LAI.png",dpi=400,bbox_inches='tight')

    model_path = \
        r"D:\Project\Research\MLP\Sun Jun 23 15_44_05 2024(lr-1e-05 epochs-20 batch_size-256)model.pth"
    output = deploy(model_path,ndvi)
    print(lai.shape, output.shape)
    c1,c2, l1,l2 = 570,600,570,600
    output = output[c1:c2, l1:l2]
    lai = lai[c1:c2, l1:l2]
    scat_fig(lai, output)


