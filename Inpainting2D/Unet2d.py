import torch
import torch.nn as nn
import os

# to remove macOS error: solution may cause crashes or silently produce incorrect results
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# convolution
def conv_3x3(in_c,out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c,out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    return conv

# crop image before concat 
def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    if delta%2 ==0:
        delta = delta//2
        return tensor[:,:,delta:tensor_size-delta, delta:tensor_size-delta]
    else:
        delta = (delta)//2
        return tensor[:,:,delta:tensor_size-delta-1, delta:tensor_size-delta-1]

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = conv_3x3(3,32)
        self.down_conv_2 = conv_3x3(32,64)
        self.down_conv_3 = conv_3x3(64,128)
        self.down_conv_4 = conv_3x3(128,256)
        self.down_conv_5 = conv_3x3(256,512)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size = 3,
            stride = 2,
            padding =1
        )

        self.up_conv_1 = conv_3x3(512,256)


        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size = 3,
            stride = 2
        )

        self.up_conv_2 = conv_3x3(256,128)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size = 3,
            stride = 2
        )

        self.up_conv_3 = conv_3x3(128,64)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size = 3,
            stride = 2
        )

        self.up_conv_4 = conv_3x3(64,32)

        self.out = nn.Conv2d(
            in_channels = 32,
            out_channels = 3,
            kernel_size = 1
        )

    def forward(self, image):
        # bs, c, h, w
        # encoder
        x1 = self.down_conv_1(image)
        print(x1.size())
        x2 = self.max_pool_2x2(x1)
        print(x2.size())
        x3 = self.down_conv_2(x2)
        print(x3.size())
        x4 = self.max_pool_2x2(x3)
        print(x4.size())
        x5 = self.down_conv_3(x4)
        print(x5.size())
        x6 = self.max_pool_2x2(x5)
        print(x6.size())
        x7 = self.down_conv_4(x6)
        print(x7.size())
        x8 = self.max_pool_2x2(x7)
        print(x8.size())
        x9 = self.down_conv_5(x8)
        print(x9.size())

        # decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7,x)
        # print(x7.size())
        print(x.size())
        print(y.size())
        x = self.up_conv_1(torch.cat([x,y],1))
        # print(x.size())
        
        x = self.up_trans_2(x)
        y = crop_img(x5,x)
        print(x.size())
        print(y.size())
        x = self.up_conv_2(torch.cat([x,y],1))

        x = self.up_trans_3(x)
        y = crop_img(x3,x)
        print(x.size())
        print(y.size())
        x = self.up_conv_3(torch.cat([x,y],1)) 
        
        x = self.up_trans_4(x)
        y = crop_img(x1,x)
        print(x.size())
        print(y.size())
        x = self.up_conv_4(torch.cat([x,y],1)) 
        
        x = self.out(x)
        print(x.size())
        return x


if __name__ == "__main__":
    image = torch.rand((1,3,256,256))
    model = UNet()
    print(model(image))