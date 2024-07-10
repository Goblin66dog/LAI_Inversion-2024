import glob
import os.path
import random

import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
import time
import numpy as np
import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import recall_score, precision_score, mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score
from model import MLP


def seed_everything(seed=114514):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ReadByBands(image_item, band):
    image_item = gdal.Open(image_item.GetSubDatasets()[band][0], gdal.GA_ReadOnly)
    image_array = np.array(image_item.ReadAsArray())
    image_array[image_array < 0 ] = 255
    # image_array = cv2.normalize(image_array, None, 0, 1,cv2.NORM_MINMAX,cv2.CV_32F)

    return image_array


def Data_List(pack_path):
    data_list = glob.glob(os.path.join(pack_path,"*.hdf"))
    return data_list

def train(device, epochs,batch_size,lr,ndvi, lai):
    batch_size_str = batch_size

    timestamp = time.time()
    readable_date_time = time.ctime(timestamp).replace(":", "_")
    writer = {
        "loss": SummaryWriter("logs"+"\\("+
                              "lr-"+str(lr)+
                              " epochs-"+str(epochs)+
                              " batch_size-"+str(batch_size)+")"+
                              str(readable_date_time)+" loss"),
        "mse": SummaryWriter("logs" + "\\(" +
                              "lr-" + str(lr) +
                              " epochs-" + str(epochs) +
                              " batch_size-" + str(batch_size) + ")" +
                              str(readable_date_time) + "mse"),
        "mae": SummaryWriter("logs" + "\\(" +
                             "lr-" + str(lr) +
                             " epochs-" + str(epochs) +
                             " batch_size-" + str(batch_size) + ")" +
                             str(readable_date_time) + "mae"),
        "mape": SummaryWriter("logs" + "\\(" +
                             "lr-" + str(lr) +
                             " epochs-" + str(epochs) +
                             " batch_size-" + str(batch_size) + ")" +
                             str(readable_date_time) + "mape"),
        "r2": SummaryWriter("logs" + "\\(" +
                             "lr-" + str(lr) +
                             " epochs-" + str(epochs) +
                             " batch_size-" + str(batch_size) + ")" +
                             str(readable_date_time) + "r2"),

    }

    #网络加载
    net = MLP(in_channels=1, out_channels=1, hidden_size=1024)
    net.to(device=device)
    net.train()

    #数据加载
    train_dataloader = DataLoader.DataLoader(ndvi, lai)
    T = torch.utils.data.DataLoader(
        dataset=train_dataloader,
        batch_size=batch_size,
        shuffle=True
    )
    V = torch.utils.data.DataLoader(
        dataset=train_dataloader,
        batch_size=1,
        shuffle=True
    )

    #loss
    loss_function = nn.SmoothL1Loss()
    best_loss = float("inf")

    #optimizer
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)

    step = 0
    for epoch in range(epochs):
        print(epoch)
        if epoch > 10:
            optimizer.param_groups[0]['lr'] = 5e-7
        for value, label in T:

            optimizer.zero_grad()

            value = value.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            prediction = net(value)

            loss = loss_function(prediction, label)
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), readable_date_time +
                           "(lr-" + str(lr) +
                           " epochs-" + str(epochs) +
                           " batch_size-" + str(batch_size_str) + ")"
                                                                  "model.pth")
            loss.backward()
            optimizer.step()
            step+=1

            writer['loss'].add_scalar("data", loss, step)


        if epoch % 1 == 0:
            valid_num = random.randint(0,1200*1200)
            breaknum=valid_num
            label = []
            prediction = []
            for val_value, val_label in V:
                net.eval()
                with torch.no_grad():
                    if valid_num == breaknum+50:
                        net.train()
                        break
                    val_value = val_value.to(device=device, dtype=torch.float32)
                    val_label = val_label.to(device=device, dtype=torch.float32)
                    val_prediction = net(val_value)
                    val_prediction = np.array(val_prediction.data.cpu()[0][0])
                    val_label = np.array(val_label.data.cpu()[0][0])
                    label.append(val_label)
                    prediction.append(val_prediction)
                    valid_num += 1
            label = np.array(label)
            prediction = np.array(prediction)
            mse = mean_squared_error(label, prediction)
            mae = mean_absolute_error(label, prediction)
            mape = mean_absolute_percentage_error(label, prediction)

            r2 = r2_score(label, prediction)
            writer['mse'].add_scalar("data", mse, step)
            writer['mae'].add_scalar("data", mae, step)
            writer['mape'].add_scalar("data", mape, step)
            writer['r2'].add_scalar("data", r2, step)
    writer["loss"].close()
    writer["mse"].close()
    writer["mae"].close()
    writer["mape"].close()
    writer["r2"].close()

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

    ndvi = ReadByBands(ndvi_item, 0)
    lai = ReadByBands(lai_item, 1)
    lai[lai>100]=255
    ndvi = cv2.normalize(ndvi, None, 0, 1,cv2.NORM_MINMAX,cv2.CV_32F)
    c1,c2, l1,l2 = 200,1200,200,1200
    ndvi = ndvi[c1:c2,l1:l2].reshape((c2-c1)*(l2-l1))
    lai  = lai[c1:c2,l1:l2].reshape((c2-c1)*(l2-l1))
    ndvi_clean = []
    lai_clean  = []
    for i in range(ndvi.shape[0]):
        if ndvi[i] != 255 and lai[i] != 255:
            ndvi_clean.append(ndvi[i])
            lai_clean.append(lai[i])
    ndvi = np.array(ndvi_clean)
    lai  = np.array(lai_clean)
    # plt.hist(lai, bins=30, color='skyblue', alpha=0.8)
    # # 设置图表属性
    # plt.title('RUNOOB hist() Test')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    #
    # # 显示图表
    # plt.show()
    seed_everything(114514)
    train("cuda",5,64, 1e-5, ndvi, lai)

# MLP

import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score
from torch.utils.tensorboard import SummaryWriter

from model import MLP
import time



# LAI<->NDVI
