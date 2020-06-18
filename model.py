import torch
import torch.nn as nn
import math
import torchvision.models
from torch.utils import model_zoo

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'MC', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.conv4_3 = LayerActivation(self.features, 22)
        self.conv5_3 = LayerActivation(self.features, 29)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, device):
        self.features(x)
        out4_3 = self.conv4_3.features.to(device)
        out5_3 = self.conv5_3.features.to(device)
        return out4_3, out5_3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(arch, cfg, batch_norm, pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls[arch])
        model.load_state_dict(state_dict)
    return model


def vgg16(pretrained=False, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, **kwargs)


class LayerActivation:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


class AuxiliaryConv(nn.Module):
    def __init__(self):
        super(AuxiliaryConv, self).__init__()
        self.conv6_7 = nn.Sequential(
            Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            Conv2d(1024, 1024, kernel_size=1)
        )

        self.conv8 = nn.Sequential(
            Conv2d(1024, 256, kernel_size=1),
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        )
        self.conv9 = nn.Sequential(
            Conv2d(512, 128, kernel_size=1),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.conv10 = nn.Sequential(
            Conv2d(256, 128, kernel_size=1),
            Conv2d(128, 256, kernel_size=3, stride=1)
        )
        self.conv11 = nn.Sequential(
            Conv2d(256, 128, kernel_size=1),
            Conv2d(128, 256, kernel_size=3, stride=1)
        )

    def forward(self, x):
        out7 = self.conv6_7(x)
        out8 = self.conv8(out7)
        out9 = self.conv9(out8)
        out10 = self.conv10(out9)
        out11 = self.conv11(out10)

        return out7, out8, out9, out10, out11


class PredictionLayerConv(nn.Module):
    def __init__(self, in_channels, dimensions, num_classes):
        super(PredictionLayerConv, self).__init__()
        self.num_classes = num_classes
        self.localization_conv = Conv2d(in_channels, dimensions * 4, kernel_size=3, padding=1)
        self.prediction_conv = Conv2d(in_channels, dimensions * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        bs = x.shape[0]
        localization = self.localization_conv(x).view(bs, -1, 4)
        prediction = self.prediction_conv(x).view(bs, -1, self.num_classes)

        return localization, prediction


class PredictionConv(nn.Module):
    def __init__(self, num_classes):
        super(PredictionConv, self).__init__()
        num_priors = {'conv_4_3': 4, 'conv_7': 6, 'conv_8': 6, 'conv_9': 6, 'conv_10': 4, 'conv_11': 4}
        self.conv_4_3 = PredictionLayerConv(512, num_priors['conv_4_3'], num_classes)
        self.conv_7 = PredictionLayerConv(1024, num_priors['conv_7'], num_classes)
        self.conv_8 = PredictionLayerConv(512, num_priors['conv_8'], num_classes)
        self.conv_9 = PredictionLayerConv(256, num_priors['conv_9'], num_classes)
        self.conv_10 = PredictionLayerConv(256, num_priors['conv_10'], num_classes)
        self.conv_11 = PredictionLayerConv(256, num_priors['conv_11'], num_classes)

    def forward(self, out_4_3, out_7, out_8, out_9, out_10, out_11):
        loc_4_3, pred_4_3 = self.conv_4_3(out_4_3)
        loc_7, pred_7 = self.conv_7(out_7)
        loc_8, pred_8 = self.conv_8(out_8)
        loc_9, pred_9 = self.conv_9(out_9)
        loc_10, pred_10 = self.conv_10(out_10)
        loc_11, pred_11 = self.conv_11(out_11)

        locs = torch.cat([loc_4_3, loc_7, loc_8, loc_9, loc_10, loc_11], dim=1)
        preds = torch.cat([pred_4_3, pred_7, pred_8, pred_9, pred_10, pred_11], dim=1)

        return locs, preds


def get_priors():
    feature_map_dimensions = {'conv4_3': 38,
                              'conv7': 19,
                              'conv8': 10,
                              'conv9': 5,
                              'conv10': 3,
                              'conv11': 1}
    prior_scales = {'conv4_3': 0.1,
                    'conv7': 0.2,
                    'conv8': 0.375,
                    'conv9': 0.55,
                    'conv10': 0.725,
                    'conv11': 0.9}
    aspect_ratios = {'conv4_3': [1, 2, 1 / 2],
                     'conv7': [1, 2, 1 / 2, 3, 1 / 3],
                     'conv8': [1, 2, 1 / 2, 3, 1 / 3],
                     'conv9': [1, 2, 1 / 2, 3, 1 / 3],
                     'conv10': [1, 2, 1 / 2],
                     'conv11': [1, 2, 1 / 2]}

    priors = []
    feature_maps = list(feature_map_dimensions.keys())

    for f, feature in enumerate(feature_maps):
        feature_dimension = feature_map_dimensions[feature]
        for i in range(feature_dimension):
            for j in range(feature_dimension):
                cx = (i + 0.5) / feature_dimension
                cy = (j + 0.5) / feature_dimension
                for ratio in aspect_ratios[feature]:
                    w = prior_scales[feature] * math.sqrt(ratio)
                    h = prior_scales[feature] / math.sqrt(ratio)
                    priors.append([cx, cy, w, h])

                    if ratio == 1:
                        try:
                            extra_prior = math.sqrt(
                                prior_scales.get(feature_maps[f]) * prior_scales.get(feature_maps[f + 1]))
                        except IndexError:
                            extra_prior = 1
                        priors.append([cx, cy, extra_prior, extra_prior])

    priors = torch.FloatTensor(priors).clamp(0, 1)
    return priors


class SSD(nn.Module):
    def __init__(self, num_classes, device):
        super(SSD, self).__init__()
        self.device = device
        self.vgg = vgg16(pretrained=True).to(device)
        self.auxiliary_conv = AuxiliaryConv().to(device)
        self.prediction_conv = PredictionConv(num_classes).to(device)
        self.priors = get_priors()

    def forward(self, x):
        out4_3, out5_3 = self.vgg(x, self.device)
        out7, out8, out9, out10, out11 = self.auxiliary_conv(out5_3)
        locs, preds = self.prediction_conv(out4_3, out7, out8, out9, out10, out11)

        return locs, preds


if __name__ == '__main__':
    img = torch.ones((2, 3, 300, 300))
    model = SSD(num_classes=4)
    o = model(img)
