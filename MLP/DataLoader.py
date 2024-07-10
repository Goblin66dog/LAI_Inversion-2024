import glob
import os.path
import random

import cv2
import numpy as np
import torch
from osgeo import gdal
from torch.utils.data.dataset import Dataset


class DataLoader(Dataset):
    def __init__(self, value, label):
        super(DataLoader).__init__()
        self.value = value
        self.label = label

    def __getitem__(self, index):
        value = self.value[index]
        label = self.label[index]
        value = value.reshape(1)
        label = label.reshape(1)
        value = torch.tensor(value, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return value, label


    def __len__(self):
        return self.label.shape[0]


if __name__ == "__main__":
    def ReadByBands(image_item, band):
        image_item = gdal.Open(image_item.GetSubDatasets()[band][0], gdal.GA_ReadOnly)
        image_array = np.array(image_item.ReadAsArray())
        image_array[image_array < 0] = 0
        image_array = cv2.normalize(image_array, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

        return image_array


    def Data_List(pack_path):
        data_list = glob.glob(os.path.join(pack_path, "*.hdf"))
        return data_list

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
    ndvi = ndvi.reshape(1200*1200)[:1200*600]
    lai  = lai.reshape(1200*1200)[:1200*600]

    train_dataloader = DataLoader(ndvi, lai)
    T = torch.utils.data.DataLoader(
        dataset=train_dataloader,
        batch_size=200,
        shuffle=True
    )
    num = 1

