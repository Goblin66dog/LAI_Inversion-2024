import glob
import os.path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from torch.utils.tensorboard import SummaryWriter
from osgeo import gdal


def scat_fig(y_true, y_pred, title):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r_squared = r2_score(y_true, y_pred)
    rmrse = rmse / (np.mean(y_true) - np.min(y_true)) if np.mean(y_true) - np.min(y_true) != 0 else np.nan
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.plot((0, 1), (0, 1), linewidth=1, transform=ax.transAxes, ls='--', c='k', label="1:1 line", alpha=0.5)
    ax.plot(y_true, y_pred, 'o', c='blue', markersize=2)
    bbox = dict(boxstyle="round", fc='1', alpha=0.)
    plt.text(0.05, 0.8, "$R^2: %.2f$\n$RMSE: %.2f$\n$RMRSE: %.2f$\n$MAE: %.2f$"
             % (r_squared, rmse, rmrse, mae),
             transform=ax.transAxes, size=7, bbox=bbox, fontdict={'family': 'Times New Roman', 'weight': 'bold'})
    ax.set_xlabel('Ground Truth', fontsize=7, fontdict={'family': 'Arial', 'size': 14, 'weight': 'bold'})
    ax.set_ylabel("Prediction", fontsize=7, fontdict={'family': 'Arial', 'size': 14, 'weight': 'bold'})
    ax.set(xlim=(0, 50), ylim=(0, 50))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 10, 20, 30, 40, 50])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}_scatter.png", dpi=400)


def ReadByBands(image_item, band):
    image_item = gdal.Open(image_item.GetSubDatasets()[band][0], gdal.GA_ReadOnly)
    image_array = np.array(image_item.ReadAsArray())
    image_array[image_array < 0] = 255
    return image_array


def Data_List(pack_path):
    data_list = glob.glob(os.path.join(pack_path, "*.hdf"))
    return data_list


def train_and_evaluate(model, model_name, ndvi, lai, c1, c2, l1, l2):
    ndvi = ndvi[c1:c2, l1:l2].reshape(-1, 1)
    lai = lai[c1:c2, l1:l2].reshape(-1)

    model.fit(ndvi, lai)

    lai_pred = model.predict(ndvi)

    scat_fig(lai, lai_pred, model_name)

    return model


if "__main__" == __name__:
    pack_path = r"C:\Users\Vitch\Desktop\Quantitative RS"
    data_list = Data_List(pack_path)
    ndvi_path = data_list[1]
    lai_path = data_list[2]

    ndvi_item = gdal.Open(ndvi_path)
    lai_item = gdal.Open(lai_path)

    ndvi = ReadByBands(ndvi_item, 0)
    lai = ReadByBands(lai_item, 1)
    lai[lai > 100] = 0
    ndvi[ndvi == 255] = 0
    ndvi = cv2.normalize(ndvi, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Training and evaluating different models
    c1, c2, l1, l2 = 570, 600, 570, 600

    # SVM
    svm_model = SVR(kernel='rbf')
    svm_model = train_and_evaluate(svm_model, "SVM", ndvi, lai, c1, c2, l1, l2)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model = train_and_evaluate(rf_model, "RandomForest", ndvi, lai, c1, c2, l1, l2)

    # Polynomial Regression
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model = train_and_evaluate(poly_model, "PolyRegression", ndvi, lai, c1, c2, l1, l2)
