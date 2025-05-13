import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import itertools
from torch.autograd import Function, Variable


def paramsInit(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.00)

def paramsInitRelu(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.01)


class Decoder(nn.Module):
    def __init__(self, output_size, input_size):
        super(Decoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(output_size, output_size//2, kernel_size=1, padding=0),
            nn.Tanh(),
            nn.Conv2d(output_size//2, output_size//4, kernel_size=1, padding=0),
            nn.Tanh(),
            nn.Conv2d(output_size // 4, output_size // 8, kernel_size=1, padding=0),
            nn.Tanh(),
            nn.Conv2d(output_size//8, input_size, kernel_size=1, padding=0),
            nn.Tanh(),
            # nn.Sigmoid(),
        )
        paramsInit(self)
    def forward(self, x):
        df2 = self.cnn(x)
        # df2=df2*2-1
        return df2


def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


class TPSGridGen(nn.Module):
    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()
        
        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)
        
        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim=1)
        
        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)
    
    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)
        
        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate


class CNN(nn.Module):
    def __init__(self, in_channels, num_output, ):  #基准点的数量，输入图像通道数
        super().__init__()
        N_patch = 24
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 25, stride=11,dilation=1, padding=12,padding_mode = 'reflect'),
            nn.MaxPool2d(5,2, padding=2),
            nn.Tanh(),
            nn.Conv2d(16, 32, 5, stride=1, padding=2,padding_mode = 'reflect'),
            nn.AvgPool2d(5,2),
            nn.MaxPool2d(5,2),
            
            # nn.Conv2d(in_channels, 16, 25, stride=11, dilation=1, padding=0),
            # nn.MaxPool2d(5, 2, padding=2),
            # nn.Tanh(),
            # nn.Conv2d(16, 32, 5, stride=1, padding=0),
            # nn.AvgPool2d(5, 2, padding=2),
            # nn.MaxPool2d(5, 2, padding=2),
            nn.AdaptiveAvgPool2d([N_patch, N_patch])
        )
        
        # for net in self.conv:
        #     if isinstance(net, nn.Conv2d):
        #         net.bias.data.zero_()
        #         net.weight.data.zero_()
        
        self.fc2 = nn.Linear(32 * N_patch * N_patch, num_output) #计算基准点使用的全连接层
        paramsInit(self)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc2(x.reshape([1, -1]))
        return x


class UnBoundedGridLocNet(nn.Module):  #With unbounded_stn, the output of locolizaition network is not squeezed--angle: int, default = 60
    def __init__(self, in_channels,grid_height, grid_width, target_control_points):
        super().__init__()
        self.cnn = CNN( in_channels, grid_height * grid_width * 2)
        
        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()
    
    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)  # 1 3 512 512
        return points.view(batch_size, -1, 2)


class BinaryActivationLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output

def map_to_normal(feature_map, mu=0, sigma=1):
    # 将特征图展平为一维张量
    flattened_feature_map = feature_map.view(-1)

    # 计算当前特征图的均值和标准差
    mean_val = flattened_feature_map.mean().detach()
    std_val = flattened_feature_map.std()#.detach()

    # 使用正态分布逆变换将特征图转换为服从指定均值和标准差的正态分布
    transformed_feature_map = (flattened_feature_map - mean_val) / std_val * sigma + mu
    # transformed_feature_map = (flattened_feature_map ) / std_val
    transformed_feature_map = transformed_feature_map.view(feature_map.shape)

    return transformed_feature_map




class FeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size, image_height, image_width):
        super().__init__()
        self.conv1_x1 = nn.Sequential(
            nn.Conv2d(input_size, output_size // 2, kernel_size=5, padding=2,padding_mode = 'reflect'),
            nn.Tanh(),
        )
        self.conv1_x2 = nn.Sequential(
            nn.Conv2d(input_size, output_size // 2, kernel_size=5, padding=2,padding_mode = 'reflect'),
            nn.Tanh(),
        )
        
        self.conv2_x1_x2 = nn.Sequential(
            # nn.Conv2d(output_size // 2, output_size // 2, kernel_size=1),
            nn.Conv2d(output_size // 2, output_size // 2, kernel_size=5,padding=2,groups=output_size // 2,padding_mode = 'reflect'),#nn.Tanh(), #
            nn.AvgPool2d(5, 1, 2),
            # nn.MaxPool2d(3, 1, 1),
            nn.Tanh(),
            #
        )

        self.conv3_x1 = nn.Sequential(
            nn.Conv2d(output_size // 2, output_size // 1, kernel_size=5, padding=2,padding_mode = 'reflect'),
            nn.Tanh(),
            
        )
        self.conv3_x2 = nn.Sequential(
            nn.Conv2d(output_size // 2, output_size // 1, kernel_size=5, padding=2,padding_mode = 'reflect'),
            nn.Tanh(),

        )
        
        self.conv4_x1_x2 = nn.Sequential(
            # nn.Conv2d(output_size // 1, output_size // 1, kernel_size=1, padding=0),
            nn.Conv2d(output_size // 1, output_size // 1, kernel_size=5, padding=2,groups=output_size // 1,padding_mode = 'reflect'),
            nn.AvgPool2d(5, 1, 2),
            # nn.MaxPool2d(3, 1, 1),
            nn.Tanh(),
        )

        paramsInit(self)
        self.conv1_x1[0].weight.data.copy_(self.conv1_x2[0].weight.data)
        self.conv1_x1[0].bias.data.copy_(self.conv1_x2[0].bias.data)
        self.conv3_x1[0].weight.data.copy_(self.conv3_x2[0].weight.data)
        self.conv3_x1[0].bias.data.copy_(self.conv3_x2[0].bias.data)
        
    def forward(self, x1,x2):
        x1 = self.conv1_x1(x1)
        x2 = self.conv1_x2(x2)
        
        x1 = self.conv2_x1_x2(x1)
        x2 = self.conv2_x1_x2(x2)
        
        x1 = self.conv3_x1(x1)
        x2 = self.conv3_x2(x2)
        
        x1 = self.conv4_x1_x2(x1)
        x2 = self.conv4_x1_x2(x2)
        
        return x1, x2


class MCAE(nn.Module):
    def __init__(self, in_img_ch, diff_ch, image_height, image_width,grid_height,grid_width):  # c, 20, c, 20
        super(MCAE, self).__init__()
        self.FeatureExtractor = FeatureExtractor(in_img_ch, diff_ch, image_height, image_width)
        self.dec_x1 = Decoder(diff_ch, in_img_ch)
        self.dec_x2 = Decoder(diff_ch, in_img_ch)
        
        self.image_height = image_height
        self.image_width = image_width
        self.diff_ch = diff_ch

        self.binary_activation = BinaryActivationLayer.apply
        
        y_grid, x_grid = torch.meshgrid(torch.arange(image_height), torch.arange(image_width))
        self.ori_flow_grid = torch.stack([x_grid.cuda(), y_grid.cuda()], dim=-1).unsqueeze(0).float()
        self.ori_flow_grid[:,:,:,0]=2*(self.ori_flow_grid[:,:,:,0]/image_width)-1
        self.ori_flow_grid[:,:,:,1]=2*(self.ori_flow_grid[:,:,:,1]/image_height)-1
        
        r1 = 0.99999  # 0.9
        r2 = 0.99999  # 0.9
        self.grid_height = grid_height
        self.grid_width = grid_width

        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)
        self.TPSGrid = TPSGridGen(image_height, image_width, target_control_points)
        target_control_points_zeros = torch.zeros_like(target_control_points)
        self.UnBoundedGridLocNet = UnBoundedGridLocNet(in_img_ch*2, grid_height, grid_width, target_control_points_zeros)
        self.target_control_points = target_control_points.unsqueeze(0).cuda()

        self.ln_1=nn.LayerNorm([self.image_height, self.image_width],elementwise_affine=False,bias=False)
        self.ln_2=nn.LayerNorm([self.image_height, self.image_width],elementwise_affine=False,bias=False)
    
    def _meshgrid(self,height,width):
            y_t = torch.linspace(-1,1, height).reshape(height,1) * torch.ones(1,width)
            x_t = torch.ones(height,1) * torch.linspace(-1,1, width).reshape(1,width)
            x_t_flat = x_t.reshape(1,1,height,width)
            y_t_flat = y_t.reshape(1,1,height,width)
            grid = torch.cat((x_t_flat,y_t_flat),1)
            return grid
    @staticmethod
    def copyNetParams(net1,net2):
        net2Generator=net2.modules()
        for m1 in net1.modules():
            m2 = net2Generator.__next__()
            if isinstance(m1, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                m2.weight.data.copy_(m1.weight.data)
                m2.bias.data.copy_(m1.bias.data)
    
    
    def forward(self, x1, x2):
        F1, F2 = self.FeatureExtractor(x1,x2)
        

        F1 = map_to_normal(F1) # 保留
        F2 = map_to_normal(F2)
        
        # F1=self.ln_1(F1)
        # F2=self.ln_2(F2)
        
        '''TPS扭曲'''
        source_ctr_bias = self.UnBoundedGridLocNet(torch.cat([x1, x2], dim=1))  # 1 3 512 512
        source_ctr_bias[0, :, 0] = 2 * (source_ctr_bias[0, :, 0] / self.image_height / self.grid_height)
        source_ctr_bias[0, :, 1] = 2 * (source_ctr_bias[0, :, 1] / self.image_width / self.grid_width)
        self.source_ctr_bias = source_ctr_bias
        
        x1_recon = self.warp_affine(self.dec_x1(F1), gridDetach=False)
        x2_recon = self.warp_affine(self.dec_x2(F2), reverse=True, gridDetach=False)
        
        return F1, F2, x1_recon, x2_recon
    
    def getGrid(self):
        grid_bias = self.TPSGrid(self.source_ctr_bias).view(1, self.image_height, self.image_width, 2)
        grid = grid_bias.detach()
        grid[..., 0] = (grid[..., 0]) / 2 * self.image_height
        grid[..., 1] = (grid[..., 1]) / 2 * self.image_width
        return grid
    
    def warp_affine(self, img, reverse=False, gridDetach=False, mode="bilinear", padding_mode="zeros"):  # reflection
        if reverse:
            source_control_points = self.target_control_points - self.source_ctr_bias
        else:
            source_control_points = self.target_control_points + self.source_ctr_bias
        source_coordinate = self.TPSGrid(source_control_points)
        batch_size = 1  #
        grid = source_coordinate.view(batch_size, self.image_height, self.image_width, 2)  ####grid
        self.grid = grid
        
        if gridDetach:
            wrp = F.grid_sample(img, self.grid.detach(), mode=mode, padding_mode=padding_mode)
        else:
            wrp = F.grid_sample(img, self.grid, mode=mode, padding_mode=padding_mode)
        return wrp
    
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 320, 256).cuda()
    y = torch.randn(1, 3, 320, 256).cuda()
    a, b, c, d = x.shape
    sz_img1 = x.shape
    Linear_num = 2 * 20 * c * d
    model =  MCAE(sz_img1[1], 32, sz_img1[2], sz_img1[3]).cuda()
    F1, F2, x1_recon, x2_recon = model(x, y)
    print(F1.shape)
    # net = clNet(20,1)

    # detach F2后，梯度只传向TPS，不传到encoder
    warped_F1 = model.warp_affine(F1, gridDetach=False)

    print(warped_F1.shape)