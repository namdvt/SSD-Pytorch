import torch
import torch.nn as nn
import math


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


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1),
            Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            Conv2d(128, 256, kernel_size=3, padding=1),
            Conv2d(256, 256, kernel_size=3, padding=1),
            Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, ceil_mode=True),
            Conv2d(256, 512, kernel_size=3, padding=1),
            Conv2d(512, 512, kernel_size=3, padding=1),
            Conv2d(512, 512, kernel_size=3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            Conv2d(512, 512, kernel_size=3, padding=1),
            Conv2d(512, 512, kernel_size=3, padding=1),
            Conv2d(512, 512, kernel_size=3, padding=1),
            Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            Conv2d(1024, 1024, kernel_size=1)
        )

    def forward(self, x):
        out_4_3 = self.conv1(x)
        out_7 = self.conv2(out_4_3)

        return out_4_3, out_7


class AuxiliaryConv(nn.Module):
    def __init__(self):
        super(AuxiliaryConv, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2d(1024, 256, kernel_size=1),
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            Conv2d(512, 128, kernel_size=1),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.conv3 = nn.Sequential(
            Conv2d(256, 128, kernel_size=1),
            Conv2d(128, 256, kernel_size=3, stride=1)
        )
        self.conv4 = nn.Sequential(
            Conv2d(256, 128, kernel_size=1),
            Conv2d(128, 256, kernel_size=3, stride=1)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)

        return out1, out2, out3, out4


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
                cx = (i+0.5)/feature_dimension
                cy = (j+0.5)/feature_dimension
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
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.vgg16 = VGG16()
        self.auxiliary_conv = AuxiliaryConv()
        self.prediction_conv = PredictionConv(num_classes)
        self.priors = get_priors()

    def forward(self, x):
        out_4_3, out_7 = self.vgg16(x)
        out_8, out_9, out_10, out_11 = self.auxiliary_conv(out_7)
        locs, preds = self.prediction_conv(out_4_3, out_7, out_8, out_9, out_10, out_11)

        return locs, preds


if __name__ == '__main__':
    img = torch.ones((2, 3, 300, 300))
    model = SSD(num_classes=90)
    o = model(img)