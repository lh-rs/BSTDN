import gc
import math
import random
import os
import time
import torch.nn.functional as F
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from matplotlib import cm
cmap = cm.get_cmap('jet', 30)
plt.set_cmap(cmap)

def getDataset(dataset:str):
    if dataset == 'g1':
        img1_path = 'MV_dataset/g1/70_1.png'
        img2_path = 'MV_dataset/g1/70_2.png'
        ref_path = 'MV_dataset/g1/ref.jpg'

    elif dataset == 'g2':
        img1_path = 'MV_dataset/g2/549_1.png'
        img2_path = 'MV_dataset/g2/549_2.png'
        ref_path = 'MV_dataset/g2/ref.jpg'

    elif dataset == 'u2':
        img1_path = 'MV_dataset/u2/25m1.png'
        img2_path = 'MV_dataset/u2/25m2.png'
        ref_path = 'MV_dataset/u2/25mref.png'

    elif dataset == 'Beijing':
        img1_path = 'MV_dataset/Beijing/beijing_A_1.jpg'
        img2_path = 'MV_dataset/Beijing/beijing_A_2.jpg'
        ref_path = 'MV_dataset/Beijing/beijing_A_gt.jpg'

    elif dataset == 'SimGloucester':
        img1_path = 'MV_dataset/SimGloucester/TPS/2tps_im1.jpg'
        img2_path = 'MV_dataset/SimGloucester/im2.jpg'
        ref_path = 'MV_dataset/SimGloucester/im3.jpg'

    elif dataset == 'SimShuguang':
        img1_path = 'MV_dataset/SimShuguang/TPS/2tps_fig1.png'
        img2_path = 'MV_dataset/SimShuguang/im2.png'
        ref_path = 'MV_dataset/SimShuguang/im3.png'

    elif dataset == 'SimSardinia':
        img1_path = 'MV_dataset/SimSardinia/TPS/2tps_im1_2.jpg'
        img2_path = 'MV_dataset/SimSardinia/im2.bmp'
        ref_path = 'MV_dataset/SimSardinia/im3.bmp'

    elif dataset == 'SimTianhe':
        img1_path = 'MV_dataset/SimTianhe/TPS/2tps_im1.jpg'
        img2_path = 'MV_dataset/SimTianhe/im2.bmp'
        ref_path = 'MV_dataset/SimTianhe/im3.bmp'
    elif dataset == 'Tianhe':
        img1_path = 'MV_dataset/SimTianhe/im1.bmp'
        img2_path = 'MV_dataset/SimTianhe/im2.bmp'
        ref_path = 'MV_dataset/SimTianhe/im3.bmp'
    return img1_path,img2_path,ref_path


def showresult(data):  # 将0-1数值转化为二值化图像（可视化过程）
    data = np.expand_dims(data, -1)
    result = np.concatenate([data, data, data], -1)
    return result


def data_Norm(data, norm=True):
    if norm:
        # data = data.detach().cpu().numpy()
        min = np.amin(data)
        max = np.amax(data)
        result = (data - min) / (max - min)  # torch.from_numpy()
    else:
        result = data
    return result


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.deterministic = True
#
def draw_flow(image, flow, step=20):
    h, w = image.shape[:2] # h w 3
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T # h w 2

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # 创建白色背景图像
    output_image = np.ones_like(image) * 255

    # 增大图像分辨率
    output_image = cv2.resize(output_image, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

    for (x1, y1), (x2, y2) in lines:
        # 绘制黑色箭头，增大箭头大小
        x1, y1, x2, y2 = x1 * 2, y1 * 2, x2 * 2, y2 * 2  # 调整坐标以匹配放大后的图像
        cv2.arrowedLine(output_image, (x1, y1), (x2, y2), (0, 0, 0), 2, tipLength=0.4)

    return output_image


def getSARData(img_path):

    img_data = np.expand_dims(data_Norm(cv2.imread(img_path)[..., 0]), -1)  # w*h*1
    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # 1*w*h tensor #permute换顺序
    img_data = img_data.unsqueeze(0)  # 1*1*w*h tensor
    print(img_data.shape)
    return img_data
def getSARData0(img_path):

    img_data = np.expand_dims(cv2.imread(img_path)[..., 0], -1)  # w*h*1
    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # 1*w*h tensor #permute换顺序
    img_data = img_data.unsqueeze(0)  # 1*1*w*h tensor
    print(img_data.shape)
    return img_data
def getRData(img_path):
    img_data =cv2.imread(img_path)  # w*h*c
    img_data= cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
    img_data = np.expand_dims(img_data, -1)
    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # c*w*h tensor
    img_data = img_data.unsqueeze(0)  # 1*c*w*h tensor

    return img_data

def resize_image(image):
    # 获取原始图像的宽度和高度
    height, width = image.shape[:2]

    # 计算新的宽度和高度（一半）
    new_width = int(width / 2)
    new_height = int(height / 2)

    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height))
    # 保存缩放后的图像
    # cv2.imwrite("resized_image.jpg", resized_image)

    return resized_image

def getRGBData(img_path):

    img_data = cv2.imread(img_path)/255 # w*h*c
    # img_data=resize_image(img_data)

    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # c*w*h tensor
    img_data = img_data.unsqueeze(0)  # 1*c*w*h tensor
    print(img_data.shape)
    return img_data
def getRGBData0(img_path):

    img_data = cv2.imread(img_path)# w*h*c
    img_data = torch.FloatTensor(img_data).permute(2, 0, 1)  # c*w*h tensor
    img_data = img_data.unsqueeze(0)  # 1*c*w*h tensor
    print(img_data.shape)
    return img_data

def getP0Data(p0_path):
    p0_data = np.expand_dims(data_Norm(cv2.imread(p0_path))[..., 0], -1)  # w*h*1
    p0_data = torch.FloatTensor(p0_data).permute(2, 0, 1)
    return p0_data

def getP02Data(p0_path):
    p0_data = np.expand_dims(cv2.imread(p0_path)[..., 0], -1)  # w*h*1
    p0_data = torch.FloatTensor(p0_data).permute(2, 0, 1)
    return p0_data

def ennoisesar(x):  # SAR, 高斯散斑噪声
    sz_x = x.shape
    noise = torch.randn(sz_x)
    x = x.cpu()
    x += x * noise
    x = torch.where(torch.BoolTensor(x > 1), torch.ones(sz_x), x)
    x = torch.where(torch.BoolTensor(x < -1), -torch.ones(sz_x), x)
    return x.detach().cuda()

def ennoisegaussian(x,var=0.001): #高斯噪声
    sz_x = x.shape
    x = x.cpu()
    noise = torch.randn(sz_x)
    nosiy = x + (var**0.5)*noise

    return nosiy.detach().cuda()

def ennoise1(x):  # SAR, 高斯散斑噪声
    sz_x = x.shape
    noise = torch.randn(sz_x)
    x = x.cpu()
    x += x * noise
    x = torch.where(torch.BoolTensor(x > 1), torch.ones(sz_x), x)
    x = torch.where(torch.BoolTensor(x < -1), -torch.ones(sz_x), x)
    return x.detach().cuda()



def liu_showresult(data,norm=False):
    from matplotlib import cm
    cmap = cm.get_cmap('jet', 30)
    plt.set_cmap(cmap)
    plt.imshow(data)
    plt.show()

def show_img_ch3(img,norm=False):  # 3*w*h, 0-1
    if norm:
        imin=img.min(-1).min(-1).reshape([-1,1,1])
        imax=img.max(-1).max(-1).reshape([-1,1,1])
        img=(img-imin)/(imax-imin+1e-15)
    img = np.transpose(img, (1, 2, 0)) * 255  # w*h*3, 0-255
    # cv2.imwrite(path, img)
    return  np.array(img,np.uint8)


def evaluate2(pred_label_data, true_label_data):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_label_data = np.reshape(pred_label_data, (1, -1))
    true_label_data = np.reshape(true_label_data, (1, -1))
    all_num = pred_label_data.shape[1]

    for i in range(all_num):
        if pred_label_data[0][i] == 255 and true_label_data[0][i] == 255:
            TP += 1
        elif pred_label_data[0][i] == 255 and true_label_data[0][i] == 0:
            FP += 1
        elif pred_label_data[0][i] == 0 and true_label_data[0][i] == 255:
            FN += 1
        else:
            TN += 1

    FP_rate = FP/(TP+FP+TN+FN)
    FN_rate = FN/(TP+FP+TN+FN)
    # OE = FP_rate+FN_rate
    OE = FP + FN
    PCC = (TP+TN)/(TP+FP+TN+FN)
    PRE = ((TP+FP)*(TP+FN))/((TP+TN+FP+FN)**2) + ((FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2)
    Kappa = (PCC-PRE)/(1-PRE)

    # 绘制PR曲线：Precision纵坐标，Recall横坐标  #PR曲线下的面积称为AP
    Precision = TP / (TP + FP) #分类准确率
    Recall = TP / (TP + FN)  #真正类率

    # 绘制roc曲线：TPR纵坐标，FPR横坐标    #ROC曲线下的面积称为AUC
    TPR = Recall  # 真正类率 TPR=TP / (TP + FN)
    FPR = FP / (TN + FP)  # 伪正类率 FPR=FP/(TN+FP)

    F1 = 2 * (Recall * Precision) / (Recall + Precision)  # a=0.5

    print('FP:' + str(FP))
    print('FN:' + str(FN))
    print('TP:' + str(TP))
    print('TN:' + str(TN))
    print('FN_rate:%.4f' %(FN_rate))
    print('FP_rate: %.4f' % (FP_rate))
    print('OE: %.4f' % (OE))
    print('PCC:%.4f' %(PCC))
    print('PRE:%.4f' %(PRE))
    print('Precision: ' + str(Precision))
    print('Recall: ' + str(Recall))
    print('Kappa: %.4f'%(Kappa))
    print('F1: %.4f' % (F1))

    return FP, FN, FN_rate, FP_rate,OE, PCC, PRE, Kappa, Precision,Recall,F1,TPR,FPR

def select_eva(pred_label_data, true_label_data):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_label_data = np.reshape(pred_label_data, (1, -1))
    true_label_data = np.reshape(true_label_data, (1, -1))
    all_num = pred_label_data.shape[1]

    for i in range(all_num):
        if pred_label_data[0][i] == 255 and true_label_data[0][i] == 255:
            TP += 1
        elif pred_label_data[0][i] == 255 and true_label_data[0][i] == 0:
            FP += 1
        elif pred_label_data[0][i] == 0 and true_label_data[0][i] == 255:
            FN += 1
        else:
            TN += 1

    Precision = TP / (TP + FP+ 1e-10)
    Recall = TP / (TP + FN)
    F = 2 * (Recall * Precision) / (Recall + Precision+ 1e-10)
    return F

def evaluate(pred_label_data, true_label_data):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_label_data = np.reshape(pred_label_data, (1, -1))
    true_label_data = np.reshape(true_label_data, (1, -1))
    all_num = pred_label_data.shape[1]

    for i in range(all_num):
        if pred_label_data[0][i] == 255 and true_label_data[0][i] == 255:
            TP += 1
        elif pred_label_data[0][i] == 255 and true_label_data[0][i] == 0:
            FP += 1
        elif pred_label_data[0][i] == 0 and true_label_data[0][i] == 255:
            FN += 1
        else:
            TN += 1

    FPR = FP/(TP+FP+TN+FN)
    FNR = FN/(TP+FP+TN+FN)
    OE = FN+FP
    OER= FNR+FPR
    PCC = (TP+TN)/(TP+FP+TN+FN)
    PRE = ((TP+FP)*(TP+FN))/((TP+TN+FP+FN)**2) + ((FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2)
    Kappa = (PCC-PRE)/(1-PRE)

    Precision = TP / (TP + FP+1e-15)
    Recall = TP / (TP + FN)
    F = 2 * (Recall * Precision) / (Recall + Precision+1e-15)
    print('FP:' + str(FP))
    print('FN:' + str(FN))
    print('OE:' + str(OE))
    print('FPR: %.4f' % (FPR))
    print('FNR: %.4f' % (FNR))
    print('OER:: %.4f' % (OER))
    print('PCC: %.4f' % (PCC))
    print('Kappa: %.4f' % (Kappa))
    print('F:  %.4f' % (F))
    return FP, FN, OE, FPR, FNR,OER, PCC, Kappa, F

def metric(img, chg_ref):

    chg_ref = np.array(chg_ref, dtype=np.float32)

    chg_ref = chg_ref / np.max(chg_ref) #[0,255]--->[0,1]
    img = img / np.max(img)

    img = np.reshape(img, [-1])
    chg_ref = np.reshape(chg_ref, [-1])

    loc1 = np.where(chg_ref == 1)[0]
    num1 = np.sum(img[loc1] == 1)
    acc_chg = np.divide(float(num1), float(np.shape(loc1)[0]))

    loc2 = np.where(chg_ref == 0)[0]
    num2 = np.sum(img[loc2] == 0)
    acc_un = np.divide(float(num2), float(np.shape(loc2)[0]))

    acc_all = np.divide(float(num1 + num2), float(np.shape(loc1)[0] + np.shape(loc2)[0]))

    loc3 = np.where(img == 1)[0]
    num3 = np.sum(chg_ref[loc3] == 1)
    acc_tp = np.divide(float(num3), float(np.shape(loc3)[0]))

    print('Accuracy of Unchanged Regions is: %.4f' % (acc_un))
    print('Accuracy of Changed Regions is:   %.4f' % (acc_chg))
    print('The True Positive ratio is:       %.4f' % (acc_tp))
    print('Accuracy of all testing sets is : %.4f' % (acc_all))
    print('')

    return acc_un, acc_chg, acc_all, acc_tp

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def save_matrix_heatmap_visual(similar_distance_map,save_change_map_dir):

    from matplotlib import cm
    cmap = cm.get_cmap('jet', 30)
    plt.set_cmap(cmap)
    plt.imsave(save_change_map_dir,similar_distance_map)

def save_path_creat(save_path):
        save_diff_path = os.path.join(save_path, 'diff/')
        if not os.path.exists(save_diff_path):
            os.makedirs(save_diff_path)
        check_dir(save_diff_path)
        save_recon_path = os.path.join(save_path, 'recon/')
        if not os.path.exists(save_recon_path):
            os.makedirs(save_recon_path)
        check_dir(save_recon_path)

        save_br_path = os.path.join(save_path, 'br/')
        if not os.path.exists(save_br_path):
            os.makedirs(save_br_path)
        check_dir(save_br_path)
        # save_heat_br_path = os.path.join(save_path, 'heatmap_br/')
        # if not os.path.exists(save_heat_br_path):
        #     os.makedirs(save_heat_br_path)
        # check_dir(save_heat_br_path)
        save_heat_diff_path = os.path.join(save_path, 'heatmap_diff/')
        if not os.path.exists(save_heat_diff_path):
            os.makedirs(save_heat_diff_path)
        check_dir(save_heat_diff_path)
        return  save_recon_path, save_diff_path, save_br_path,save_heat_diff_path

def save_img_ch1(img, path):  # w*h, 0-1
    img = np.expand_dims(img, -1)*255   # w*h*1, 0-255
    img = np.concatenate([img, img, img], -1)  # w*h*3, 0-255
    cv2.imwrite(path, img.astype(np.uint8))


def save_img_ch3(img, path,norm=False):  # 3*w*h, 0-1
    if norm:
        imin = img.min(-1).min(-1).reshape([-1, 1, 1])
        imax = img.max(-1).max(-1).reshape([-1, 1, 1])
        img=(img-imin)/(imax-imin+1e-15)
    img = np.transpose(img, (1, 2, 0)) * 255  # w*h*3, 0-255
    cv2.imwrite(path, img)

# 将图像二值化
def pic2binary(data):
    datasize = data.shape
    datax = np.zeros(datasize)
    for i in range(datasize[0]):
        for j in range(datasize[1]):
            if data[i][j] >= 128:
                datax[i][j] = 1
    return datax


def bilary2pic(data):  # 将0-1数值转化为二值化图像（可视化过程）
    datasize = data.shape
    for i in range(datasize[0]):
        for j in range(datasize[1]):
            if data[i][j] == 1:
                data[i][j] = 255
    data = np.expand_dims(data, -1)
    result = np.concatenate([data, data, data], -1)
    return result

def minmaxscaler(data):
    data = data.detach().cpu().numpy()
    min = np.amin(data)
    max = np.amax(data)
    result = torch.from_numpy((data - min) / (max - min))
    return result



def lr_schedule(epoch):
    lr = 0.01
    if epoch > 25:
        lr = 0.01
    elif epoch > 20:
        lr = 0.005
    else:
        if epoch > 15:
            lr = 0.001
        else:
            if epoch > 10:
                lr = 0.005
            else:
                if epoch > 5:
                    lr = 0.01
    print('Learning rate: ', lr)
    return lr

def otsu(data, num=400, get_bcm=False):
    """
    generate binary change map based on otsu
    :param data: cluster data
    :param num: intensity number
    :param get_bcm: bool, get bcm or not
    :return:
        binary change map
        selected threshold
    """
    max_value = np.max(data)
    min_value = np.min(data)

    total_num = data.shape[1]
    step_value = (max_value - min_value) / num
    value = min_value + step_value
    best_threshold = min_value
    best_inter_class_var = 0
    while value <= max_value:
        data_1 = data[data <= value]
        data_2 = data[data > value]
        if data_1.shape[0] == 0 or data_2.shape[0] == 0:
            value += step_value
            continue
        w1 = data_1.shape[0] / total_num
        w2 = data_2.shape[0] / total_num

        mean_1 = data_1.mean()
        mean_2 = data_2.mean()

        inter_class_var = w1 * w2 * np.power((mean_1 - mean_2), 2)
        if best_inter_class_var < inter_class_var:
            best_inter_class_var = inter_class_var
            best_threshold = value
        value += step_value
    if get_bcm:
        bwp = np.zeros(data.shape)
        bwp[data <= best_threshold] = 0
        bwp[data > best_threshold] = 255
        print('otsu is done')
        return bwp, best_threshold
    else:
        print('otsu is done')
        print('otsu==', best_threshold)
        return best_threshold

def evaluate(pred_label_data, true_label_data):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_label_data = np.reshape(pred_label_data, (1, -1))
    true_label_data = np.reshape(true_label_data, (1, -1))
    all_num = pred_label_data.shape[1]

    for i in range(all_num):
        if pred_label_data[0][i] == 255 and true_label_data[0][i] == 255:
            TP += 1
        elif pred_label_data[0][i] == 255 and true_label_data[0][i] == 0:
            FP += 1
        elif pred_label_data[0][i] == 0 and true_label_data[0][i] == 255:
            FN += 1
        else:
            TN += 1

    FPR = FP/(TP+FP+TN+FN)
    FNR = FN/(TP+FP+TN+FN)
    OE = FN+FP
    OER= FNR+FPR
    PCC = (TP+TN)/(TP+FP+TN+FN)
    PRE = ((TP+FP)*(TP+FN))/((TP+TN+FP+FN)**2) + ((FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2)
    Kappa = (PCC-PRE)/(1-PRE+ 1e-15)
    Precision = TP / (TP + FP + 1e-15)
    Recall = TP / (TP + FN+ 1e-15)
    F = 2 * (Recall * Precision) / (Recall + Precision+ 1e-15)

    print('FP:' + str(FP))
    print('FN:' + str(FN))
    print('OE:' + str(OE))
    print('FPR: %.4f' % (FPR))
    print('FNR: %.4f' % (FNR))
    print('OER:: %.4f' % (OER))
    print('PCC: %.4f' % (PCC))
    print('Kappa: %.4f' % (Kappa))
    print('F:  %.4f' % (F))
    return FP, FN, OE, FPR, FNR,OER, PCC, Kappa, F

def select_eva(pred_label_data, true_label_data):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_label_data = np.reshape(pred_label_data, (1, -1))
    true_label_data = np.reshape(true_label_data, (1, -1))
    all_num = pred_label_data.shape[1]

    for i in range(all_num):
        if pred_label_data[0][i] == 255 and true_label_data[0][i] == 255:
            TP += 1
        elif pred_label_data[0][i] == 255 and true_label_data[0][i] == 0:
            FP += 1
        elif pred_label_data[0][i] == 0 and true_label_data[0][i] == 255:
            FN += 1
        else:
            TN += 1

    Precision = TP / (TP + FP+ 1e-10)
    Recall = TP / (TP + FN)
    F = 2 * (Recall * Precision) / (Recall + Precision+ 1e-10)
    return F





from sklearn.decomposition import PCA
def applyPCA(data,n_components=3):
    newX = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=n_components, whiten=True)
    newX = pca.fit_transform(newX)
    data = np.reshape(newX, (data.shape[0], data.shape[1], -1))
    return data


from skimage import transform
from lib.cnn_feature import Feature_Extracter
import plotmatch
class WarpCNN():
    def __init__(self,img_scales=[0.5,1,2],RESIDUAL_THRESHOLD=20):
        self.img_scales=img_scales
        self.cnn_feature_extract = Feature_Extracter()
        '''变化前后图 之间 匹配点的坐标差阈值'''
        self._RESIDUAL_THRESHOLD = RESIDUAL_THRESHOLD
        
        
    def warp(self,srcImg,dstImg,nfeatures=-1):
        start = time.perf_counter()
        start0 = time.perf_counter()
        
        kps_left, sco_left, des_left, dense_fea_list_1 = self.cnn_feature_extract(srcImg, img_scales=self.img_scales,
                                                                             nfeatures=nfeatures)
        kps_right, sco_right, des_right, dense_fea_list_2 = self.cnn_feature_extract(dstImg, img_scales=self.img_scales,
                                                                                nfeatures=nfeatures)
        
        
        print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % (
        (time.perf_counter() - start), len(kps_left), len(kps_right)))
        
        '''显示特征差异图'''
        fig, ([ax1, ax2, ], [ax3, ax4]) = plt.subplots(2, 2)
        diff = torch.sum(torch.sqrt((dense_fea_list_1[0] - dense_fea_list_2[0]) ** 2), dim=1, keepdim=False).squeeze()
        diff = diff.detach().cpu().numpy()
        ax1.imshow(diff)
        
        diff = torch.sum(torch.sqrt((dense_fea_list_1[1] - dense_fea_list_2[1]) ** 2), dim=1, keepdim=False).squeeze()
        diff = diff.detach().cpu().numpy()
        ax2.imshow(diff)
        
        diff = torch.sum(torch.sqrt((dense_fea_list_1[2] - dense_fea_list_2[2]) ** 2), dim=1, keepdim=False).squeeze()
        diff = diff.detach().cpu().numpy()
        ax3.imshow(diff)
        
        # dense_fea_1=torch.cat(dense_fea_list_1,dim=1)
        # dense_fea_2=torch.cat(dense_fea_list_2,dim=1)
        
        dense_fea_1 = dense_fea_list_1[0] + dense_fea_list_1[1] + 1.44 * dense_fea_list_1[2]
        dense_fea_2 = dense_fea_list_2[0] + dense_fea_list_2[1] + 1.44 * dense_fea_list_2[2]
        
        # dense_fea_1=dense_fea_1
        # dense_fea_2=dense_fea_2
        
        # B = 1  # batch size
        # # 初始化旋转角度和平移量
        # angle = 0 / 180 * math.pi
        # shift_x = 1
        # shift_y = 1
        # # 创建一个坐标变换矩阵
        # transform_matrix = torch.tensor([
        #     [math.cos(angle), math.sin(-angle), 0],
        #     [math.sin(angle), math.cos(angle), 0]])
        # # 将坐标变换矩阵的shape从[2,3]转化为[1,2,3]，并重复在第0维B次，最终shape为[B,2,3]
        # transform_matrix = transform_matrix.unsqueeze(0).repeat(B, 1, 1)
        #
        # grid = F.affine_grid(transform_matrix,  # 旋转变换矩阵
        #                      dense_fea_2.shape).cuda()  # 变换后的tensor的shape(与输入tensor相同)
        #
        # dense_fea_2 = F.grid_sample(dense_fea_2,  # 输入tensor，shape为[B,C,W,H]
        #                        grid,  # 上一步输出的gird,shape为[B,C,W,H]
        #                        mode='nearest')  # 一些图像填充方法，这里我用的是最近邻

        
        diff = torch.sum(torch.sqrt((dense_fea_1 - dense_fea_2) ** 2), dim=1, keepdim=False).squeeze()
        diff = diff.detach().cpu().numpy()
        ax4.imshow(diff)
        
        fig.tight_layout()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.show()
        
        del dense_fea_list_1
        del dense_fea_list_2
        # del dense_fea_1,diff
        torch.cuda.empty_cache()
        gc.collect()
        
        start = time.perf_counter()
        # Flann特征匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # trees=5
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # flann = cv2.BFMatcher() # 更换为BFMatcher
        matches = flann.knnMatch(des_left, des_right, k=2)
        
        
        '''测试'''
        # # 准备一个空的掩膜来绘制好的匹配
        # mask_matches = [[0, 0] for i in range(len(matches))]
        #
        # # 向掩膜中添加数据
        # for i, (m, n) in enumerate(matches):
        #     if m.distance < 0.7 * n.distance:
        #         mask_matches[i] = [1, 0]
        #
        # img_matches = cv2.drawMatchesKnn(srcImg, kps_left, dstImg, kps_right, matches, None,
        #                                  matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        #                                  matchesMask=mask_matches, flags=0)
        #
        # cv2.imshow("FLANN", img_matches)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        
        
        goodMatch = []
        locations_1_to_use = []
        locations_2_to_use = []
        
        # 匹配对筛选
        min_dist = 1000
        max_dist = 0
        disdif_avg = 0
        # 统计平均距离差
        for m, n in matches:
            disdif_avg += n.distance - m.distance
        disdif_avg = disdif_avg / len(matches)
        
        for m, n in matches:
            # 自适应阈值
            if n.distance > m.distance + disdif_avg:
                goodMatch.append(m)
                p2 = cv2.KeyPoint(kps_right[m.trainIdx][0], kps_right[m.trainIdx][1], 1)
                p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
                locations_1_to_use.append([p1.pt[0], p1.pt[1]])
                locations_2_to_use.append([p2.pt[0], p2.pt[1]])
        # goodMatch = sorted(goodMatch, key=lambda x: x.distance)
        print('match num is %d' % len(goodMatch))
        locations_1_to_use = np.array(locations_1_to_use)
        locations_2_to_use = np.array(locations_2_to_use)
        
        # Perform geometric verification using RANSAC. 离群值剔除,重要!
        # slope=(locations_1_to_use[:,0]-locations_2_to_use[:,0])/(locations_1_to_use[:,1]-locations_2_to_use[:,1])
        
        _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                                    transform.AffineTransform,
                                    # transform.ProjectiveTransform,
                                    min_samples=4, #3
                                    residual_threshold=self._RESIDUAL_THRESHOLD,
                                    max_trials=10000)
        
        print('Found %d inliers' % sum(inliers))
        # inliers=~inliers | inliers
        # inliers=~inliers
        
        inlier_idxs = np.nonzero(inliers)[0]
        # 最终匹配结果
        matches = np.column_stack((inlier_idxs, inlier_idxs))
        print('whole time is %6.3f' % (time.perf_counter() - start0))
        
        # Visualize correspondences, and save to file.
        # 1 绘制匹配连线
        plt.rcParams['savefig.dpi'] = 500  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率
        plt.rcParams['figure.figsize'] = (4.0, 3.0)  # 设置figure_size尺寸
        fig, ax = plt.subplots()
        plotmatch.plot_matches(
            ax,
            srcImg,
            dstImg,
            locations_1_to_use,
            locations_2_to_use,
            np.column_stack((inlier_idxs, inlier_idxs)),
            plot_matche_points=False,
            matchline=True,
            matchlinewidth=0.2)
        ax.axis('off')
        ax.set_title('')
        fig.tight_layout()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        
        plt.show()
        
        ## 扭曲校正测试
        matched_points_1 = locations_1_to_use[inlier_idxs, :]
        matched_points_2 = locations_2_to_use[inlier_idxs, :]
        
        # M, mask = cv2.findHomography(matched_points_1, matched_points_2, cv2.RANSAC, 3)
        M, valid_point = cv2.findHomography(matched_points_1, matched_points_2, 0, )
        
        warped_img1 = cv2.warpPerspective(srcImg, M, (srcImg.shape[1], srcImg.shape[0]))
        mask = np.ones((srcImg.shape[0], srcImg.shape[1]), dtype=np.uint8)
        mask = cv2.warpPerspective(mask, M, (srcImg.shape[1], srcImg.shape[0]))
        mask = np.expand_dims(mask, -1)  # .astype(float)
        
        # warped_img1=tps_warp(matched_points_1,matched_points_2,srcImg,srcImg.shape,smooth=0.01)
        
        
        fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2)
        SHOWIMG = np.sqrt((warped_img1 / 255 - dstImg / 255) ** 2) * mask
        
        ax1.imshow(np.array(SHOWIMG * 255, np.uint8))
        
        # ax2.imshow(mask)
        # 校正后的特征差异图
        dense_fea_1 = dense_fea_1[0].permute([1, 2, 0]).detach().cpu().numpy()
        dense_fea_2 = dense_fea_2[0].permute([1, 2, 0]).detach().cpu().numpy()
        dense_fea_1 = cv2.warpPerspective(dense_fea_1, M, (srcImg.shape[1], srcImg.shape[0]))
        diff = np.sum(np.sqrt((dense_fea_1 - dense_fea_2) ** 2), axis=-1, keepdims=False).squeeze() #* mask.squeeze()
        ax2.imshow(diff)
        
        ax3.imshow(dstImg * mask)
        ax4.imshow(warped_img1)
        
        # 设置紧凑布局
        fig.tight_layout()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.show()
        
        '''最终匹配上的点'''
        valid_point = valid_point.squeeze(-1)
        # matched_points=matched_points_1[valid_point] # img1已经校正，不再需要坐标
        matched_points = matched_points_2[np.array(valid_point, dtype=bool)]
        
        # 坐标数据
        matched_grid = np.zeros(srcImg.shape[:2])
        for p in matched_points:
            matched_grid[int(p[1]), int(p[0])] = 1
        
        return warped_img1, dstImg* mask, mask, matched_grid, dense_fea_1,dense_fea_2* mask
    
    def delModel(self):
        self.cnn_feature_extract.delModel()
        del self.cnn_feature_extract
        gc.collect()

def normalizeData(data, type="norm"):
    from sklearn import preprocessing

    height, width, bands = data.shape  # 原始高光谱数据的三个维度
    data = np.reshape(data, [height * width, bands])

    if type == "norm":
        # # 标准归一化
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)  ####################
        data = np.reshape(data, [height, width, bands])

    if type == "minmax":
        minMax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        data = minMax.fit_transform(data)  ####################
        data = np.reshape(data, [height, width, bands])

    if type == "none":
        data = np.reshape(data, [height, width, bands])
    return data

from scipy.interpolate import Rbf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def tps_warp(Y, T, Y_image, out_shape,smooth=1):
    Y_height, Y_width = Y_image.shape[:2]
    T_height, T_width = out_shape[:2]
    
    i_func = Rbf(T[:, 0], T[:, 1], Y[:, 0], function='thin-plate',smooth=smooth)
    # i_func = Rbf(T[:, 0], T[:, 1], Y[:, 0], function='multiquadric')
    j_func = Rbf(T[:, 0], T[:, 1], Y[:, 1], function='thin-plate',smooth=smooth)
    # j_func = Rbf(T[:, 0], T[:, 1], Y[:, 1], function='multiquadric')

    iT, jT = np.mgrid[:T_height, :T_width]
    iT = iT.flatten()
    jT = jT.flatten()
    iY = np.int_(i_func(iT, jT))
    jY = np.int_(j_func(iT, jT))

    keep = np.logical_and(iY>=0, jY>=0)
    keep = np.logical_and(keep, iY<Y_height)
    keep = np.logical_and(keep, jY<Y_width)
    iY, jY, iT, jT = iY[keep], jY[keep], iT[keep], jT[keep]

    out_image = np.zeros(out_shape, dtype='uint8')
    out_image[iT, jT, :] = Y_image[iY, jY, :]

    return out_image
def TwoPercentLinear(image, th:int=2, max_out=255, min_out=0):
    b, g, r = cv2.split(image)#分开三个波段
    def gray_process(gray, maxout = max_out, minout = min_out):
        high_value = np.percentile(gray, 100-th)#取得98%直方图处对应灰度
        low_value = np.percentile(gray, th)#同理
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        processed_gray = ((truncated_gray - low_value)/(high_value - low_value)) * (maxout - minout)#线性拉伸嘛
        return processed_gray
    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = cv2.merge((b_p, g_p, r_p))#合并处理后的三个波段
    return np.uint8(result)

def np_map_to_normal(feature_map, mu=0, sigma=1):
    # 将特征图展平为一维张量
    flattened_feature_map = feature_map.reshape([-1])

    # 计算当前特征图的均值和标准差
    mean_val = flattened_feature_map.mean()
    std_val = flattened_feature_map.std()

    # 使用正态分布逆变换将特征图转换为服从指定均值和标准差的正态分布
    transformed_feature_map = (flattened_feature_map - mean_val) / std_val * sigma + mu
    transformed_feature_map = transformed_feature_map.reshape(feature_map.shape)

    return transformed_feature_map

def getcor(x, y):  # 1*1*w*h*c
    # x = x[0, 0, :, :, :]
    # y = y[0, 0, :, :, :]
    x = x[0].permute([1, 2, 0])
    y = y[0].permute([1, 2, 0])
    
    x_reducemean = x - torch.mean(x, dim=2, keepdim=True)
    y_reducemean = y - torch.mean(y, dim=2, keepdim=True)
    # print("x_red: ", x_reducemean.shape)
    numerator = torch.sum(torch.mul(x_reducemean, y_reducemean),dim=2)
    # print("num: ", numerator.shape)
    x_no = torch.norm(x_reducemean, p=2, dim=2)
    y_no = torch.norm(y_reducemean, p=2, dim=2)
    # print("x_no.shape: ", x_no.shape)
    denominator = torch.mul(x_no, y_no)
    # print("den: ", denominator.shape)
    corrcoef = numerator / denominator
    return corrcoef

import torch

def nr(im1, im2, k):
    ylen, xlen = im1.shape
    ratio = torch.zeros(ylen, xlen)
    nrmap = torch.zeros(ylen, xlen)

    im1 = torch.tensor(im1, dtype=torch.float32)
    im2 = torch.tensor(im2, dtype=torch.float32)

    for j in range(ylen):
        for i in range(xlen):
            if im1[j, i] > im2[j, i]:
                ratio[j, i] = (im2[j, i] + 0.001) / (im1[j, i] + 0.001)
            elif im1[j, i] < im2[j, i]:
                ratio[j, i] = (im1[j, i] + 0.001) / (im2[j, i] + 0.001)
            else:
                ratio[j, i] = 1

    for j in range(int(1 + (k - 1) / 2), int(ylen - (k - 1) / 2)):
        for i in range(int(1 + (k - 1) / 2), int(xlen - (k - 1) / 2)):
            u = 0
            diat = 0
            smin = 0
            smax = 0

            im_se1 = im1[j - int((k - 1) / 2):j + int((k - 1) / 2) + 1, i - int((k - 1) / 2):i + int((k - 1) / 2) + 1]
            im_se2 = im2[j - int((k - 1) / 2):j + int((k - 1) / 2) + 1, i - int((k - 1) / 2):i + int((k - 1) / 2) + 1]
            rat_se = ratio[j - int((k - 1) / 2):j + int((k - 1) / 2) + 1, i - int((k - 1) / 2):i + int((k - 1) / 2) + 1]

            smin = (im_se1 * (im_se1 <= im_se2)).sum().item() + (im_se2 * (im_se1 > im_se2)).sum().item()
            smax = (im_se1 * (im_se1 >= im_se2)).sum().item() + (im_se2 * (im_se1 < im_se2)).sum().item()

            u = rat_se.mean().item()
            diat = rat_se.var().item()
            lmd = (diat + 0.001) / (u + 0.001)

            if lmd > 1:
                lmd = 1

            if smax == 0:
                nrmap[j, i] = lmd * ratio[j, i] + (1 - lmd)
            else:
                nrmap[j, i] = lmd * ratio[j, i] + (1 - lmd) * smin / smax

    # 处理边上的像素
    tmp = nrmap[int(1 + (k - 1) / 2):int(ylen - (k - 1) / 2), int(1 + (k - 1) / 2):int(xlen - (k - 1) / 2)]
    u = tmp.mean().item()
    nrmap[0:int(1 + (k - 1) / 2), :] = u
    nrmap[int(ylen - (k - 1) / 2):ylen, :] = u
    nrmap[:, 0:int(1 + (k - 1) / 2)] = u
    nrmap[:, int(xlen - (k - 1) / 2):xlen] = u

    return nrmap

import torch

def nr_multichannel(im1, im2, k):
    ylen, xlen, channels = im1.shape
    ratio = torch.zeros(ylen, xlen, channels)
    nrmap = torch.zeros(ylen, xlen, channels)

    im1 = torch.tensor(im1, dtype=torch.float32)
    im2 = torch.tensor(im2, dtype=torch.float32)

    for j in range(ylen):
        for i in range(xlen):
            for c in range(channels):
                if im1[j, i, c] > im2[j, i, c]:
                    ratio[j, i, c] = (im2[j, i, c] + 0.001) / (im1[j, i, c] + 0.001)
                elif im1[j, i, c] < im2[j, i, c]:
                    ratio[j, i, c] = (im1[j, i, c] + 0.001) / (im2[j, i, c] + 0.001)
                else:
                    ratio[j, i, c] = 1

    for j in range(int(1 + (k - 1) / 2), int(ylen - (k - 1) / 2)):
        for i in range(int(1 + (k - 1) / 2), int(xlen - (k - 1) / 2)):
            for c in range(channels):
                u = 0
                diat = 0
                smin = 0
                smax = 0

                im_se1 = im1[j - int((k - 1) / 2):j + int((k - 1) / 2) + 1,
                            i - int((k - 1) / 2):i + int((k - 1) / 2) + 1, c]
                im_se2 = im2[j - int((k - 1) / 2):j + int((k - 1) / 2) + 1,
                            i - int((k - 1) / 2):i + int((k - 1) / 2) + 1, c]
                rat_se = ratio[j - int((k - 1) / 2):j + int((k - 1) / 2) + 1,
                               i - int((k - 1) / 2):i + int((k - 1) / 2) + 1, c]

                smin = (im_se1 * (im_se1 <= im_se2)).sum().item() + (im_se2 * (im_se1 > im_se2)).sum().item()
                smax = (im_se1 * (im_se1 >= im_se2)).sum().item() + (im_se2 * (im_se1 < im_se2)).sum().item()

                u = rat_se.mean().item()
                diat = rat_se.var().item()
                lmd = (diat + 0.001) / (u + 0.001)

                if lmd > 1:
                    lmd = 1

                if smax == 0:
                    nrmap[j, i, c] = lmd * ratio[j, i, c] + (1 - lmd)
                else:
                    nrmap[j, i, c] = lmd * ratio[j, i, c] + (1 - lmd) * smin / smax

    # 处理边上的像素
    tmp = nrmap[int(1 + (k - 1) / 2):int(ylen - (k - 1) / 2), int(1 + (k - 1) / 2):int(xlen - (k - 1) / 2), :]
    u = tmp.mean().item()
    nrmap[0:int(1 + (k - 1) / 2), :, :] = u
    nrmap[int(ylen - (k - 1) / 2):ylen, :, :] = u
    nrmap[:, 0:int(1 + (k - 1) / 2), :] = u
    nrmap[:, int(xlen - (k - 1) / 2):xlen, :] = u

    return nrmap

import torch

def nr_multichannel_parallel(im1, im2, k):
    ylen, xlen, channels = im1.shape
    ratio = torch.zeros(ylen, xlen, channels)
    nrmap = torch.zeros(ylen, xlen, channels)

    im1 = torch.tensor(im1, dtype=torch.float32)
    im2 = torch.tensor(im2, dtype=torch.float32)

    for c in range(channels):
        im1_channel = im1[:, :, c]
        im2_channel = im2[:, :, c]

        mask1 = im1_channel > im2_channel
        mask2 = im1_channel < im2_channel
        mask3 = ~mask1 & ~mask2

        ratio[:, :, c] = torch.where(mask1, (im2_channel + 0.001) / (im1_channel + 0.001),
                                      torch.where(mask2, (im1_channel + 0.001) / (im2_channel + 0.001), torch.ones_like(im2_channel)))

    for c in range(channels):
        for j in range(int(1 + (k - 1) / 2), int(ylen - (k - 1) / 2)):
            for i in range(int(1 + (k - 1) / 2), int(xlen - (k - 1) / 2)):
                im_se1 = im1[j - int((k - 1) / 2):j + int((k - 1) / 2) + 1,
                            i - int((k - 1) / 2):i + int((k - 1) / 2) + 1, c]
                im_se2 = im2[j - int((k - 1) / 2):j + int((k - 1) / 2) + 1,
                            i - int((k - 1) / 2):i + int((k - 1) / 2) + 1, c]
                rat_se = ratio[j - int((k - 1) / 2):j + int((k - 1) / 2) + 1,
                               i - int((k - 1) / 2):i + int((k - 1) / 2) + 1, c]

                smin = (im_se1 * (im_se1 <= im_se2)).sum().item() + (im_se2 * (im_se1 > im_se2)).sum().item()
                smax = (im_se1 * (im_se1 >= im_se2)).sum().item() + (im_se2 * (im_se1 < im_se2)).sum().item()

                u = rat_se.mean().item()
                diat = rat_se.var().item()
                lmd = (diat + 0.001) / (u + 0.001)

                if lmd > 1:
                    lmd = 1

                if smax == 0:
                    nrmap[j, i, c] = lmd * ratio[j, i, c] + (1 - lmd)
                else:
                    nrmap[j, i, c] = lmd * ratio[j, i, c] + (1 - lmd) * smin / smax

    # 处理边上的像素
    tmp = nrmap[int(1 + (k - 1) / 2):int(ylen - (k - 1) / 2), int(1 + (k - 1) / 2):int(xlen - (k - 1) / 2), :]
    u = tmp.mean().item()
    nrmap[0:int(1 + (k - 1) / 2), :, :] = u
    nrmap[int(ylen - (k - 1) / 2):ylen, :, :] = u
    nrmap[:, 0:int(1 + (k - 1) / 2), :] = u
    nrmap[:, int(xlen - (k - 1) / 2):xlen, :] = u

    return nrmap

# 示例用法
# nrmap_result = nr_multichannel_parallel(im1_tensor, im2_tensor, k_value)




def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.deterministic = True