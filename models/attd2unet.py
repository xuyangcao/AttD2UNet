import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ResidualCBAMLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=5, padding=2, stride=1, dilation=1, groups=1, bias=False, act_func='relu', reduction_ratio=None, pool_types=['avg', 'max'], **kwargs):
        super().__init__()
        # channel attention module
        self.gate_channels = in_channels
        if reduction_ratio is not None:
            self.reduction_ratio = reduction_ratio
        else:
            self.reduction_ratio = self.gate_channels // 6
        self.pool_types = pool_types
        self.mlp = nn.Sequential(
                nn.Linear(self.gate_channels, self.gate_channels // self.reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(self.gate_channels // self.reduction_ratio, self.gate_channels)
                )

        # spacial attention module
        self.in_channels = in_channels
        self.act_func = act_func
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.spacial = nn.Conv3d(2, 1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = nn.ReLU()
        self.relu = nn.ReLU()

    def forward(self, x):
        # channel attention module
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                avg_pool = avg_pool.squeeze()
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                max_pool = max_pool.squeeze()
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x_channel = x * scale

        # spacial attention module
        x_compress = torch.cat((torch.max(x_channel, dim=1, keepdim=True)[0], torch.mean(x_channel, dim=1, keepdim=True)), dim=1)
        x_out = self.act(self.spacial(x_compress))
        scale = torch.sigmoid(x_out)
        return self.relu(x + x_channel * scale)

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, padding, dilation):
        super(_DenseLayer, self).__init__()
        # modules for bottle neck layer
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, 
                                            kernel_size=1, stride=1, bias=False))
        # modules for dense  layer
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, 
                                            #kernel_size=3, stride=1, padding=padding, 
                                            kernel_size=5, stride=1, padding=2*padding, 
                                            dilation=dilation, bias=False))
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))

        return bottleneck_output

    def forward(self, inputs):
        prev_features = inputs
        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, 
                drop_rate, output_stride, use_dilation=True):
        super(_DenseBlock, self).__init__()
        if output_stride == 2:
            dilations = [1, 2, 3, 1, 2, 3]
        elif output_stride == 4:
            dilations = [1, 2, 3, 1, 2, 3]
        else:
            dilations = [1, 1, 1, 1, 1, 1]

        if use_dilation:
            num_dilations = len(dilations)
            # layers wiout atrous convolution
            for i in range(num_layers - num_dilations):
                layer = _DenseLayer(
                        num_input_features + i * growth_rate,
                        growth_rate=growth_rate,
                        bn_size=bn_size,
                        drop_rate=drop_rate,
                        padding=1,
                        dilation=1
                        )
                self.add_module('denselayer%d' % (i + 1), layer)
            # layers with atrous convolution
            for i in range(num_layers - num_dilations, num_layers):
                layer = _DenseLayer(
                        num_input_features + i * growth_rate,
                        growth_rate=growth_rate,
                        bn_size=bn_size,
                        drop_rate=drop_rate,
                        padding=dilations[i - num_layers + num_dilations],
                        dilation=dilations[i - num_layers + num_dilations]
                        )
                self.add_module('atrouslayer%d' % (i + 1), layer)
        else:
            for i in range(num_layers):
                layer = _DenseLayer(
                        num_input_features + i * growth_rate,
                        growth_rate=growth_rate,
                        bn_size=bn_size,
                        drop_rate=drop_rate,
                        padding=1,
                        dilation=1
                        )
                self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features= layer(features)
            features.append(new_features)

        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, is_pool=True):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if is_pool:
            self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class UpSampleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, skip_channels=0, scale_factor=(2, 2, 2)):
        super(UpSampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.up = nn.functional.interpolate

        self.conv = nn.Conv3d(in_planes+skip_channels, out_planes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        if skip_channels != 0:
            self.cbam = ResidualCBAMLayer(skip_channels)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, out=None):
        #x = self.up(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
        x = self.up(x, scale_factor=self.scale_factor, mode='nearest')
        if out is not None:
            #print('out.shape: ', out.shape)
            #print('x.shape: ', x.shape)
            out = self.cbam(out)
            x = torch.cat([x, out], 1)
        x = self.relu(self.bn(self.conv(x)))

        return x

class OutputBlock(nn.Module):
    def __init__(self, inChans):
        super(OutputBlock, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(2)
        self.relu1 = nn.ReLU(inplace=True) 
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        return out


class AtrousDenseNet(nn.Module):
    """
    DenseNet 3D with atrous convolution

    Args:
    growth_rate (int): how many filters to add each layer ('k' in paper)
    block_config (list of 4 ints): how many layers in each pooling block
    num_init_features (int): the number of filters to learn in the first convolution layer
    bn_size (int): multiplicative factor for number of bottle neck layers. (i.e. bn_size * k features in the bottleneck layer)
    drop_rate (float): dropout rate after each dense layer
    num_classes (int): number of segmentation class
    dilations (list of 6 ints): atrous rate in atrous convolution 
    """

    def __init__(self, in_planes=1, out_planes=2, growth_rate=6, block_config=(6, 6, 6, 6), num_init_features=16, bn_size=4, drop_rate=0., use_dilation=True):
        super(AtrousDenseNet, self).__init__()

        # Input convolution 
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_planes, num_init_features, kernel_size=5, 
                                stride=1, padding=2, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1))
            ]))

        # Dense Blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features, 
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    output_stride=2 ** (i + 1),
                    use_dilation=use_dilation,
                    )
            #print(2 ** (i + 1))
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                if i == len(block_config) - 2:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2, is_pool=False)
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
                else:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))


        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        out = self.features(x)

        return out


class AttD2UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, drop_rate=0.3, skip_connetcion=True):
        super(AttD2UNet, self).__init__()
        self.skip_connetcion = skip_connetcion
        features = AtrousDenseNet(in_channels, num_classes, drop_rate=drop_rate).features

        # building atrous dense unet
        self.input_block = features[:4]
        self.dense_block_1 = features[4]
        self.transition_block_1 = features[5]
        self.dense_block_2 = features[6]
        self.transition_block_2 = features[7]
        self.dense_block_3 = features[8]
        self.transition_block_3 = features[9]
        self.dense_block_4 = features[10]
        self.transition_block_4 = features[11]

        self.up_1 = UpSampleBlock(32, 16, scale_factor=(1, 2, 2)) 
        if skip_connetcion:
            self.up_2 = UpSampleBlock(64, 32, 52)
            self.up_3 = UpSampleBlock(69, 64, 62)
        else:
            self.up_2 = UpSampleBlock(64, 32)
            self.up_3 = UpSampleBlock(69, 64)

        self.output_block = OutputBlock(16) 

    def forward(self, x):
        #print('x.shape', x.shape)
        x_64 = self.input_block(x)
        x_64 = self.dense_block_1(x_64)
        #print('x_64.shape', x_64.shape)
        x_32 = self.transition_block_1(x_64)
        x_32 = self.dense_block_2(x_32)
        #print('x_32.shape', x_32.shape)
        x_16 = self.transition_block_2(x_32)
        x_16 = self.dense_block_3(x_16)
        #print('x_16.shape', x_16.shape)
        x_8 = self.transition_block_3(x_16)
        x_8 = self.dense_block_4(x_8)
        #print('x_8.shape', x_8.shape)
        out = self.transition_block_4(x_8)
        #print('out.shape', out.shape)
        if self.skip_connetcion:
            out = self.up_3(out, x_32)
            out = self.up_2(out, x_64)
        else:
            out = self.up_3(out)
            out = self.up_2(out)
        out = self.up_1(out)
        out = self.output_block(out)
        return out
