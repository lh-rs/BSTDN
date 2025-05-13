import skimage
import torch
import xlwt
import datetime
from CD_utils import *
from models.Networks import *
import torch.nn.functional as F
from matplotlib import cm

if __name__ == '__main__':
    T1 = time.perf_counter()
    cmap = cm.get_cmap('jet', 30)
    plt.set_cmap(cmap)

    '''超参数等'''
    date_object = datetime.date.today()
    Channels = 32

    grid_size = 2

    for dataset in ['Beijing','u2','g1','g2','SimGloucester','SimShuguang','SimSardinia','SimTianhe']:
    # for dataset in ['Beijing','u2','g1','g2']:

        setup_seed(2025)
        imgfile1, imgfile2, ref_path = getDataset(dataset)

        '''读取双时相图像'''
        start = time.perf_counter()
        image1 = cv2.imread(imgfile1)
        image2 = cv2.imread(imgfile2)
        if image2.shape[0] != image1.shape[0] or image2.shape[1] != image1.shape[1]:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_NEAREST)
        if len(image1.shape) <= 2: image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
        if len(image2.shape) <= 2: image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

        image1 = np.array(image1, dtype=float) / 255
        image2 = np.array(image2, dtype=float) / 255
        image1 = image1 * 2 - 1
        image2 = image2 * 2 - 1

        img1_data = torch.FloatTensor(image1).permute(2, 0, 1).unsqueeze(0)  # c*w*h tensor
        img2_data = torch.FloatTensor(image2).permute(2, 0, 1).unsqueeze(0)  # c*w*h tensor

        ref_data = cv2.imread(ref_path)[..., 0]
        print('max',np.max(ref_data))
        sz_img1 = img1_data.shape

        N = sz_img1[1] * sz_img1[2] * sz_img1[3]
        img1_data = img1_data.cuda()
        img2_data = img2_data.cuda()

        print('image size is', image1.shape)
        print('read image time is %6.3f' % (time.perf_counter() - start))

        '''保存路径'''
        result_path0 = 'result_mv_output2/' + dataset + '/grid_size_' + str(grid_size) + "x"+ str(grid_size)+'/'

        result_path = os.path.join(result_path0)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        check_dir(result_path)

        '''构建模型'''
        model = MCAE(sz_img1[1], Channels, sz_img1[2], sz_img1[3], grid_size, grid_size).cuda()

        MSE = torch.nn.MSELoss().cuda()
        ReconLoss = torch.nn.MSELoss().cuda()

        epochs = 50
        iters = 50

        i = 1
        wk = xlwt.Workbook()
        ws = wk.add_sheet('analysis')
        ws.write(0, 0, "epoch")
        ws.write(0, 1, "FP")
        ws.write(0, 2, "FN")
        ws.write(0, 3, "OE")
        ws.write(0, 4, "PCC")
        ws.write(0, 5, "Kappa")
        ws.write(0, 6, "F1")
        ws.write(0, 9, "acc_un")
        ws.write(0, 10, "acc_chg")
        ws.write(0, 11, "acc_all")
        ws.write(0, 12, "acc_tp")

        """正式训练"""
        TPS_Params = list(map(id, model.UnBoundedGridLocNet.parameters()))
        base_params = filter(lambda p: id(p) not in TPS_Params, model.parameters())
        
        base_params = list(base_params)
        
        optimizer = torch.optim.Adamax(
            [
                {"params": model.UnBoundedGridLocNet.parameters(), "lr": 0.0001},
                {"params": base_params, "lr": 0.001},
            ],
            lr=0.001, weight_decay=0.00001
        )  #

        # diff_list=[] # 所有训练阶段的diff
        grad_params_list = list(model.FeatureExtractor.parameters())

        param_grad_prev = {param: torch.randn_like(param.data) for param in grad_params_list}

        loss_best = 9999999
        for epoch in range(epochs):
            for iter in range(iters):
                F1, F2, x1_recon, x2_recon = model(img1_data, img2_data)

                mask = torch.ones([F1.shape[0], 1, F1.shape[2], F1.shape[3]], requires_grad=False).cuda()
                mask2 = model.warp_affine(mask+0.1).detach().int()
                mask1 = model.warp_affine(mask+0.1, reverse=True).detach().int()

                # detach F2后，梯度只传向TPS，不传到encoder
                warped_F1 = model.warp_affine(F1, gridDetach=False)
                diff = torch.sqrt(torch.mean((warped_F1 - F2.detach()) ** 2, dim=1,keepdim=True)) #* mask2


                allMask=(1-torch.max_pool2d(1-(mask2 * mask1).float(),5, stride=1, padding=2)).int() # 去除边缘影响
                
                diff = diff * allMask.unsqueeze(1)
                diff_test = (map_to_normal(diff) + diff) / 2 if dataset in ['','Beijing','g2','SimSardinia','SimGloucester','u2'] else diff #
                Loss_fdm = F.l1_loss(diff_test, torch.zeros_like(diff_test).cuda())
                
                diff=diff.squeeze(1)
                
                reconLoss_1 = ReconLoss(x1_recon , img2_data )#* mask2
                reconLoss_2 = ReconLoss(x2_recon  , img1_data  )#* mask1
                Loss_Reconstruct = (reconLoss_1 + reconLoss_2) / 2
                
                loss =1 * Loss_Reconstruct + 0.5 * Loss_fdm

                optimizer.zero_grad()
                loss.backward()
                
                if dataset in ['SimShuguang','g2','SimTianhe','SimGloucester','u2','Beijing']:

                    for param in grad_params_list:
                        if param.grad is not None:
                            # 获取当前梯度
                            current_grad = param.grad.data
                            # # 获取上一步的梯度
                            prev_grad = param_grad_prev[param]
                            param_idx_1 = param_grad_prev[param] >= 0
                            param_idx_2 = param_grad_prev[param] < 0
                            # 若上一步中的参数梯度大于0，则当前步中的梯度也要大于0，否则置零
                            current_grad[param_idx_1] = torch.where(current_grad[param_idx_1] >= 0, current_grad[param_idx_1], torch.zeros_like(current_grad[param_idx_1]))
                            # 若上一步中的梯度为负值，则当前步中的梯度也需要为负值，否则置零
                            current_grad[param_idx_2] = torch.where(current_grad[param_idx_2] <0, current_grad[param_idx_2], torch.zeros_like(current_grad[param_idx_2]))

                        
                optimizer.step()

                if (iter+1) % 49 == 0:
                    warped_img1 = model.warp_affine(img1_data).detach()
                    warped_img2 = model.warp_affine(img2_data).detach()
                    output_image = np.zeros([sz_img1[2], sz_img1[3], 3])
                    flow = model.getGrid()[0].detach().cpu().numpy()
                    flow_img = draw_flow(output_image, flow)
                    flow_th = np.sum(flow ** 2, axis=-1, keepdims=False)
                    # 保存图片
                    fig1 = plt.figure()
                    plt.imshow(flow_img)
                    fig1.savefig(result_path + str(epoch) + '_' + "flow_img.jpg", dpi=300, bbox_inches='tight')
                    fig2 = plt.figure()
                    plt.imshow(flow_th)
                    fig2.savefig(result_path + str(epoch) + '_' + "flow_th.jpg", dpi=600, bbox_inches='tight')
   
                
                info = ''
                if loss_best >= loss :
                    loss_best = loss
                    curr_reconLoss_best=Loss_Reconstruct
                    curr_CLLoss_best=0.5*Loss_fdm
                    F1_best, F2_best = F1, F2
                    diff_best = diff
                    x1_recon_best = x1_recon
                    x2_recon_best = x2_recon
                    info = '\tsave best results....'

            '''生成差异图'''
            img1 = diff_best.squeeze().cpu().detach().numpy()

            img1 = (img1 - img1.min()) / (img1.max() - img1.min())

            if dataset in ['SimTianhe']:
                th = skimage.filters.threshold_multiotsu(img1, classes=4)[-1]
            else:
                th = skimage.filters.threshold_yen(img1)
            

            bestresult = np.where(img1 > th, 255, 0)
            cv2.imwrite(result_path + str(epoch) + '_' + 'bestresult.jpg', showresult(bestresult))
            print('=======================',dataset,epoch,' loss_best=',loss_best.item(),'recon_loss=',curr_reconLoss_best.item(),
                  'CL_loss=',curr_CLLoss_best.item(),)
            print('th:', th)

            warped_image1 = warped_img1.squeeze().cpu().detach().numpy()
            warped_image2 = warped_img2.squeeze().cpu().detach().numpy()
            
            save_img_ch1(allMask.squeeze().cpu().detach().numpy(), result_path + 'mask_{}.jpg'.format(epoch))
            save_img_ch3(data_Norm(warped_image1, norm=True), result_path + 'x1_warp_{}.jpg'.format(epoch))
            save_img_ch3(data_Norm(warped_image2, norm=True), result_path + 'x2_warp_{}.jpg'.format(epoch))
            save_img_ch3(x1_recon.squeeze().cpu().detach().numpy(), result_path + 'recon_1_{}.jpg'.format(epoch), norm=True)
            save_img_ch3(x2_recon.squeeze().cpu().detach().numpy(), result_path + 'recon_2_{}.jpg'.format(epoch), norm=True)
            img1 = diff.squeeze().cpu().detach().numpy()

            save_matrix_heatmap_visual(img1, result_path + 'heat_diff_{}_{}.jpg'.format(epoch,iter))
            save_img_ch1(data_Norm(img1, norm=True), result_path + 'diff_{}_{}.jpg'.format(epoch,iter))


            FP, FN, OE, FPR, FNR, OER, PCC, Kappa, F1 = evaluate(bestresult, ref_data)
            acc_un, acc_chg, acc_all, acc_tp = metric(bestresult, ref_data)

            ws.write(i, 0, epoch)
            ws.write(i, 1, FP)
            ws.write(i, 2, FN)
            ws.write(i, 3, OE)
            ws.write(i, 4, PCC)
            ws.write(i, 5, Kappa)
            ws.write(i, 6, F1)

            ws.write(i, 9, acc_un)
            ws.write(i, 10, acc_chg)
            ws.write(i, 11, acc_all)
            ws.write(i, 12, acc_tp)

            i += 1

            wk.save(result_path + '0_'+dataset + '_Eval.xls')

            # diff_list.append(diff.detach().cpu().numpy())


        '''存储diff_list'''
        # diff_list = np.concatenate(diff_list,axis=0).transpose([1,2,0])
        # hdf5storage.savemat(result_path + 'diff_list.mat', {'diff': diff_list})
    T2 = time.perf_counter()
    Sum_times = (T2 - T1)
    print("Sum_times:", Sum_times)