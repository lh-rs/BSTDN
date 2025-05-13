import torch
import numpy as np
from user_cd.dataload import getSARData, getRGBData, getP0Data, getSARData0, liu_showresult, getRGBData0
# from models.MCAE_CBAM_Cross import MCAE, clNet
# from models.MCAE_CBAM import MCAE, clNet
from models.MCAE import MCAE, clNet
from user_cd.utils import save_img_ch1, save_img_ch3, minmaxscaler, otsu
import cv2
from user_cd.evaluate import select_eva, evaluate, metric
import itertools
import torch.nn as nn
from models.loss import ContrastiveLoss, Smooth_contrastive
import xlwt
import matplotlib.pyplot as plt
import os
import random
import time
import datetime
def save_matrix_heatmap_visual(similar_distance_map,save_change_map_dir):
    from matplotlib import cm
    cmap = cm.get_cmap('jet', 30)
    plt.set_cmap(cmap)
    plt.imsave(save_change_map_dir,similar_distance_map)

def showresult(data):  # 将0-1数值转化为二值化图像（可视化过程）
    data = np.expand_dims(data, -1)
    result = np.concatenate([data, data, data], -1)
    return result

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
def data_Norm(data, norm=True):
    if norm:
        # data = data.detach().cpu().numpy()
        min = np.amin(data)
        max = np.amax(data)
        result = (data - min) / (max - min)  # torch.from_numpy()
    return result

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    date_object = datetime.date.today()
    lossfunction = 'CL'

    # for dataset in ['Beijing','u2','g1', 'g2']:
    for dataset in ['Beijing','u2','g1', 'g2','SimGloucester','SimShuguang','SimTianhe','SimSardinia']:
        if dataset == 'Beijing':
            setup_seed(2025)
            m = 0.1
            img1_path = 'result_mv_output2/Beijing/grid_size_2x2/x1_warp_49.jpg'
            img2_path = 'MV_dataset/Beijing/beijing_A_2.jpg'
            ref_path = 'MV_dataset/Beijing/beijing_A_gt.jpg'
            pse_path = 'result_mv_output2/Beijing/grid_size_2x2/49_bestresult.jpg'
            mask_path = 'result_mv_output2/Beijing/grid_size_2x2/mask_49.jpg'
        elif dataset == 'u2':
            setup_seed(2025)
            m = 0.01
            img1_path = 'result_mv_output2/u2/grid_size_2x2/x1_warp_49.jpg'
            pse_path = 'result_mv_output2/u2/grid_size_2x2/49_bestresult.jpg'
            img2_path = 'MV_dataset/u2/25m2.png'
            ref_path = 'MV_dataset/u2/25mref.png'
            mask_path = 'result_mv_output2/u2/grid_size_2x2/mask_49.jpg'
        elif dataset == 'g1':
            setup_seed(2025)
            m = 0.05
            img1_path = 'result_mv_output2/g1/grid_size_2x2/x1_warp_49.jpg'
            img2_path = 'MV_dataset/g1/70_2.png'
            ref_path = 'MV_dataset/g1/ref.jpg'
            pse_path = 'result_mv_output2/g1/grid_size_2x2/49_bestresult.jpg'
            mask_path = 'result_mv_output2/g1/grid_size_2x2/mask_49.jpg'
        elif dataset == 'g2':
            m = 0.2
            setup_seed(2019)
            img1_path = 'result_mv_output2/g2/grid_size_2x2/x1_warp_49.jpg'
            img2_path = 'MV_dataset/g2/549_2.png'
            ref_path = 'MV_dataset/g2/ref.jpg'
            pse_path = 'result_mv_output2/g2/grid_size_2x2/49_bestresult.jpg'
            mask_path = 'result_mv_output2/g2/grid_size_2x2/mask_49.jpg'

        elif dataset == 'SimGloucester':
            setup_seed(2025)
            img1_path = 'result_mv_output2/SimGloucester/grid_size_2x2/x1_warp_49.jpg'
            img2_path = 'MV_dataset/SimGloucester/im2.jpg'
            ref_path = 'MV_dataset/SimGloucester/im3.jpg'
            pse_path = 'result_mv_output2/SimGloucester/grid_size_2x2/49_bestresult.jpg'
            mask_path = 'result_mv_output2/SimGloucester/grid_size_2x2/mask_49.jpg'
            m = 0.3
        elif dataset == 'SimShuguang':
            setup_seed(2025)
            img1_path = 'result_mv_output2/SimShuguang/grid_size_2x2/x1_warp_21.jpg'
            img2_path = 'MV_dataset/SimShuguang/im2.png'
            ref_path = 'MV_dataset/SimShuguang/im3.png'
            pse_path = 'result_mv_output2/SimShuguang/grid_size_2x2/21_bestresult.jpg'
            mask_path = 'result_mv_output2/SimShuguang/grid_size_2x2/mask_49.jpg'
            m = 0.2

        elif dataset == 'SimSardinia':
            setup_seed(2025)
            img1_path = 'result_mv_output2/SimSardinia/grid_size_2x2/x1_warp_49.jpg'
            img2_path = 'MV_dataset/SimSardinia/im2.bmp'
            ref_path = 'MV_dataset/SimSardinia/im3.bmp'
            pse_path = 'result_mv_output2/SimSardinia/grid_size_2x2/49_bestresult.jpg'
            mask_path = 'result_mv_output2/SimSardinia/grid_size_2x2/mask_49.jpg'
            m = 0.1
        elif dataset == 'SimTianhe':
            setup_seed(2025)
            img1_path = 'result_mv_output2/SimTianhe/grid_size_2x2/x1_warp_29.jpg'
            img2_path = 'MV_dataset/SimTianhe/im2.bmp'
            ref_path = 'MV_dataset/SimTianhe/im3.bmp'
            pse_path = 'result_mv_output2/SimTianhe/grid_size_2x2/29_bestresult.jpg'
            mask_path = 'result_mv_output2/SimTianhe/grid_size_2x2/mask_49.jpg'
            m = 0.2

        result_path0 = ('res_RMs') + '/' + dataset + '_' + str(m)  + '/'

        result_path = os.path.join(result_path0)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        check_dir(result_path)

        Channels = 20

        img1_data = getRGBData(img1_path)
        img2_data = getRGBData(img2_path)
        mask_data = getRGBData(mask_path)

        img1_data=img1_data*mask_data
        img2_data=img2_data*mask_data
        mask_data=mask_data[:,0].cuda()
    
        cva_ref_data = cv2.imread(pse_path)[..., 0]

        ref_data = cv2.imread(ref_path)[..., 0]
        sz_img1 = img1_data.shape
        sz_img2 = img2_data.shape

        N = sz_img1[1] * sz_img1[2] * sz_img1[3]
        img1_data = img1_data.cuda()
        img2_data = img2_data.cuda()

        # Randomly initializing
        pcx = torch.rand(sz_img1[1], sz_img1[2], sz_img1[3]).cuda()
        print('Randomly initializing Pc: {}'.format(torch.sum(pcx) / pcx.numel()))

        model = MCAE(sz_img1[1], Channels, sz_img2[1], Channels).cuda()
        net = clNet(2 * Channels, 1).cuda()

        BCE = torch.nn.MSELoss().cuda()
        SCL = Smooth_contrastive().cuda()
        CL = ContrastiveLoss().cuda()

        optimizer = torch.optim.RMSprop([{'params': model.parameters(), 'lr': 0.0001},
                                         {'params': net.parameters(), 'lr': 0.001}],
                                        )  # weight_decay=0

        epochs = 50
        iters = 100

        i = 1
        wk = xlwt.Workbook()
        ws = wk.add_sheet('analysis')
        ws.write(0, 0, "epoch")
        ws.write(0, 1, "m")
        ws.write(0, 2, "FP")
        ws.write(0, 3, "FN")
        ws.write(0, 4, "OE")
        ws.write(0, 5, "PCC")
        ws.write(0, 6, "Kappa")
        ws.write(0, 7, "F")

        model.train()
        for epoch in range(epochs):
            if epoch == 0:
                pse_path = pse_path
            else:
                pse_path = result_path + str(epoch - 1) + '_' + 'bestresult.jpg'
            pse_data = getP0Data(pse_path).unsqueeze(0).cuda()

            loss_best = 9999999
            for iter in range(iters):
                F1_1, f1_2, f1_3, F2_1, f2_2, f2_3 = model(img1_data, img2_data)
                diff = torch.sqrt(torch.sum((F1_1 - F2_1) ** 2, dim=1))
                diff = diff*mask_data
                unchanged_weithts = pse_data.sum() / (sz_img1[2] * sz_img1[3])
                changed_weithts = 1 - unchanged_weithts


                weight = pse_data * 1 + (1 - pse_data) * m
                if lossfunction == 'CL':
                    loss_1 = CL(diff, pcx, unchanged_weithts)

                elif lossfunction == 'SCL':
                    loss_1 = SCL(diff, pcx, m)

                pos_fea = torch.sqrt((F1_1 - F2_1) ** 2 + 1e-15)

                pre_data = net(pos_fea)

                loss_2 = torch.mean((pre_data - pse_data) ** 2 * weight)

                info = ''
                if iter <= 30:
                    loss = loss_2 + 10
                else:
                    loss = 1*loss_1 + 5 * loss_2  # 5


                if loss_best > loss:
                    loss_best = loss
                    pre_data_best = pre_data
                    diff_best = diff
                    info = '\tsave best results....'
                else:
                    info = ' '

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pre_data = pre_data_best.squeeze(0).detach()
            pcx = pre_data

            img1 = diff.squeeze().cpu().detach().numpy()
            img2 = pcx.squeeze().cpu().detach().numpy()

            last_pcx_1 = pcx if epoch == 0 else last_pcx_1 + pcx
            img2_1 = (last_pcx_1 / (epoch + 1)).squeeze().detach().cpu().numpy()

            save_img_ch1(data_Norm(img1, norm=True), result_path + 'diff_{}.jpg'.format(epoch))
            save_img_ch1(img2_1, result_path + 'pc_{}.jpg'.format(epoch))
            save_matrix_heatmap_visual(data_Norm(img1, norm=True), result_path + 'heat_diff_{}.jpg'.format(epoch))

            Fbefore = 0
            thresh = np.array(list(range(80, 200))) / 255

            best_th = 0
            for th in thresh:
                seclect_result = np.where(img2_1 > th, 255, 0)
                last_pc = cv2.imread(pse_path)[...,0]

                F0 = 1 / np.sum((seclect_result - last_pc) ** 2 + 1e-15)
                if F0 > Fbefore:
                    cv2.imwrite(result_path + str(epoch) + '_' + 'bestresult.jpg', showresult(seclect_result))
                    Fbefore = F0
                    best_th = th
            print(dataset,epoch,' best_th=======================', best_th)

            bestresult = cv2.imread(result_path + str(epoch) + '_' + 'bestresult.jpg')[..., 0]

            FP, FN, OE, FPR, FNR, OER, PCC, Kappa, F = evaluate(bestresult, ref_data)
            acc_un, acc_chg, acc_all, acc_tp = metric(bestresult, ref_data)

            ws.write(i, 0, epoch)
            ws.write(i, 1, m)
            ws.write(i, 2, FP)
            ws.write(i, 3, FN)
            ws.write(i, 4, OE)
            ws.write(i, 5, PCC)
            ws.write(i, 6, Kappa)
            ws.write(i, 7, F)


            i += 1

            wk.save(result_path + dataset + '_' + lossfunction +'_Eval.xls')
