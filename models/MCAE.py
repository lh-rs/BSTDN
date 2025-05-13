import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn


def paramsInit(net):
    if isinstance(net, nn.Conv2d):
        nn.init.xavier_uniform_(net.weight.data)
        nn.init.constant_(net.bias.data, 0.00)
    if isinstance(net, nn.Linear):
        nn.init.xavier_uniform_(net.weight.data)
        nn.init.constant_(net.bias.data, 0.01)
    # if isinstance(net, nn.Conv2d):
    #     nn.init.xavier_uniform_(net.weight.data)
    #     nn.init.constant_(net.bias.data, 0.0)
    # elif isinstance(net, nn.BatchNorm2d):
    #     net.weight.data.fill_(1)
    #     net.bias.data.zero_()
    # elif isinstance(net, nn.Linear):
    #     net.weight.data.normal_(0, 0.001)
    #     net.bias.data.zero_()

class Encoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()

        self.forward_pass1 = nn.Sequential(
            # nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
            
            nn.Conv2d(input_size, output_size, kernel_size=1, padding=0),#nn.Tanh(),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1,groups=output_size),
            # nn.MaxPool2d(3, 1, 1),
            nn.AvgPool2d(3, 1, 1),
            
            nn.Tanh(),
           
        )
        self.forward_pass2 = nn.Sequential(
            # nn.Conv2d(input_size, output_size, kernel_size=5, padding=2),
            
            nn.Conv2d(input_size, output_size, kernel_size=1, padding=0),# nn.Tanh(),
            nn.Conv2d(output_size, output_size, kernel_size=5, padding=2, groups=output_size),
            # nn.MaxPool2d(5, 1, 2),
            nn.AvgPool2d(3, 1, 1),
            nn.Tanh(),
        )
        self.forward_pass3= nn.Sequential(
            nn.Conv2d(2*output_size, 2*output_size, kernel_size=1, padding=0),
            nn.Tanh(),
            nn.Conv2d(2*output_size, 2*output_size, kernel_size=1, padding=0),
            # nn.Tanh(),
            # nn.Conv2d(2*output_size, 2*output_size, kernel_size=1, padding=0),
            # nn.Sigmoid()
            # nn.AvgPool2d(3,1,1),
            # nn.Softplus(),
            nn.Tanh(),
            # nn.Sigmoid()
        )
        paramsInit(self)

    def forward(self, x):

        f2 = self.forward_pass1(x)
        f3 = self.forward_pass2(x)
        F = torch.cat([f2,f3], dim=1)
        F = self.forward_pass3(F)
        # F =  self.Res_group( F )

        return F, f2, f3


class clNet(nn.Module):
    def __init__(self, input_size,output_size):
        super(clNet, self).__init__()

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(input_size, input_size, kernel_size=3, padding=2,dilation=2),
        #     nn.Sigmoid()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(input_size, input_size, kernel_size=3, padding=2,dilation=2),
        #     nn.Sigmoid()
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(input_size, input_size, kernel_size=3, padding=2,dilation=2),
        #     nn.Sigmoid(),
        #     nn.Conv2d(input_size, input_size, kernel_size=3, padding=2,dilation=2),
        #     nn.Sigmoid()
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, input_size, kernel_size=3,padding=2,dilation=2),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_size, input_size, kernel_size=1,padding=0),
            nn.Tanh()
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(input_size, input_size, kernel_size=1),
        #     nn.Tanh(),
        #     nn.Conv2d(input_size, input_size, kernel_size=1),
        #     nn.Tanh()
        # )
        self.conv4 = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=1),
            nn.Sigmoid()
        )
        paramsInit(self)

    def forward(self, x):
        cov1 = self.conv1(x)
        cov2 = self.conv2(cov1)
        # cov3 = self.conv3(cov2)
        result = self.conv4(cov2)
        return result


class Decoder(nn.Module):

    def __init__(self, output_size,input_size):
        super(Decoder, self).__init__()

        self.backward_pass2 = nn.Sequential(
            nn.ConvTranspose2d(2*output_size, output_size, kernel_size=1),
            nn.Tanh(),
            nn.ConvTranspose2d(output_size, input_size, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.backward_pass3 = nn.Sequential(
            nn.ConvTranspose2d(2*output_size, output_size, kernel_size=1),
            nn.Tanh(),
            nn.ConvTranspose2d(output_size, input_size, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):

        df2 = self.backward_pass2(x)
        df3 = self.backward_pass3(x)

        return df2, df3


class MCAE(nn.Module):
    def __init__(self, in_ch1, out_ch1, in_ch2, out_ch2):  # c, 20, c, 20
        super(MCAE, self).__init__()
        self.enc_x1 = Encoder(in_ch1, out_ch1)
        self.enc_x2 = Encoder(in_ch2, out_ch2)
        self.norm = nn.Softmax2d()
        self.dec_x1 = Decoder(out_ch1,in_ch1)
        self.dec_x2 = Decoder(out_ch2, in_ch2)

        paramsInit(self)
        
    def forward(self, x1, x2, pretraining=False):

        F1_1, f1_2, f1_3 = self.enc_x1(x1)
        F2_1, f2_2, f2_3 = self.enc_x2(x2)


        if pretraining:
            x1_recon2, x1_recon3 = self.dec_x1(F1_1)
            x2_recon2, x2_recon3 = self.dec_x2(F2_1)

            return x1_recon2, x1_recon3, x2_recon2, x2_recon3

        else:
            return F1_1, f1_2, f1_3, F2_1, f2_2, f2_3


if __name__ == '__main__':
    x = torch.randn(1,20,235,233)
    net = clNet(20,1)
    y= net(x)
    print(y.shape)

