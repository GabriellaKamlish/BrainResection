import torch.nn as nn


class nnUnetConvBlock(nn.Module):
    """
    The basic convolution building block of nnUnet.
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.nonlin = nn.LeakyReLU
        #self.dropout = nn.Dropout3d
        self.dropout = None
        #print('Initialising model without dropout.')
        #self.norm = nn.BatchNorm3d
        self.norm = nn.GroupNorm
        self.conv = nn.Conv3d

        self.nonlin_args = {'negative_slope': 1e-2, 'inplace': True}
        self.dropout_args = {'p': 0.5, 'inplace': True}
        #self.norm_args = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        self.norm_args = {'eps': 1e-5, 'affine': True}
        self.conv_args = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.conv = self.conv(self.input_channels, self.output_channels, **self.conv_args)
        if self.dropout is not None and self.dropout_args['p'] is not None and self.dropout_args['p'] > 0:
            self.dropout = self.dropout(**self.dropout_args)
        else:
            self.dropout = None
        #self.norm = self.norm(num_features=self.output_channels, **self.norm_args)
        #print('Output channels: {} BatchNorm groups: {}'.format(self.output_channels, self.output_channels//4))
        self.norm = self.norm(num_channels=self.output_channels, num_groups=self.output_channels//2, **self.norm_args)
        self.nonlin = self.nonlin(**self.nonlin_args)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.norm(x)
        return self.nonlin(x)


class nnUnetConvBlockStack(nn.Module):
    """
    Concatenates multiple nnUnetConvBlocks.
    """
    def __init__(self, input_channels, output_channels, num_blocks):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stack = nn.Sequential(*([nnUnetConvBlock(input_channels, output_channels)]
                                   +[nnUnetConvBlock(output_channels, output_channels) for _ in range(num_blocks-1)]))


    def forward(self, x):
        return self.stack(x)


class GifNetEncoder(nn.Module):
    def __init__(
            self,
            input_channels=1,
            base_num_channels=30,
            num_pool=3,
            ):
        super().__init__()
        self.upsample_mode = 'trilinear'
        self.pool = nn.MaxPool3d

        self.downsample_path_convs = []
        self.downsample_path_pooling = []

        output_channels = base_num_channels
        for level in range(num_pool):
            # Add two convolution blocks
            self.downsample_path_convs.append(nnUnetConvBlockStack(input_channels, output_channels, 2))
            # Add pooling
            self.downsample_path_pooling.append(self.pool([2,2,2]))
            # Calculate input/output channels for next level
            input_channels = output_channels
            output_channels *= 2

        # now the 'bottleneck'
        final_num_channels = self.downsample_path_convs[-1].output_channels
        stack_io = nnUnetConvBlockStack(input_channels, output_channels, 1)
        stack_of = nnUnetConvBlockStack(output_channels, final_num_channels, 1)
        self.downsample_path_convs.append(nn.Sequential(stack_io, stack_of))

        # register modules
        self.downsample_path_convs = nn.ModuleList(self.downsample_path_convs)
        self.downsample_path_pooling = nn.ModuleList(self.downsample_path_pooling)

        # run weight initialisation
        from torch.nn.init import kaiming_normal_, normal_
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
                kaiming_normal_(module.weight, a=1e-2, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias,0)

    def forward(self, x, average=True):
        features = x
        for level in range(len(self.downsample_path_convs) - 1):
            conv_layer = self.downsample_path_convs[level]
            pooling_layer = self.downsample_path_pooling[level]
            features = conv_layer(features)
            features = pooling_layer(features)
        if average:
            features = features.mean(dim=(-3, -2, -1))
        return features
