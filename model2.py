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
        num_priors = 4
        self.conv4 = PredictionLayerConv(64, num_priors, num_classes)
        self.conv5 = PredictionLayerConv(48, num_priors, num_classes)
        self.conv6 = PredictionLayerConv(48, num_priors, num_classes)
        self.conv7 = PredictionLayerConv(32, num_priors, num_classes)

    def forward(self, out4, out5, out6, out7):
        loc4, pred4 = self.conv4(out4)
        loc5, pred5 = self.conv5(out5)
        loc6, pred6 = self.conv6(out6)
        loc7, pred7 = self.conv7(out7)

        locs = torch.cat([loc4, loc5, loc6, loc7], dim=1)
        preds = torch.cat([pred4, pred5, pred6, pred7], dim=1)

        return locs, preds


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv2d(48, 64, kernel_size=3, stride=2)
        self.conv4 = Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv5 = Conv2d(64, 48, kernel_size=3, stride=2)
        self.conv6 = Conv2d(48, 48, kernel_size=3, stride=2)
        self.conv7 = Conv2d(48, 32, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out4 = self.conv4(x)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        out7 = self.conv7(out6)

        return out4, out5, out6, out7


def get_priors():
    feature_map_dimensions = {'conv4': 31,
                              'conv5': 15,
                              'conv6': 7,
                              'conv7': 3}
    prior_scales = {'conv4': 0.1,
                    'conv5': 0.3,
                    'conv6': 0.6,
                    'conv7': 0.9}
    aspect_ratios = [1, 2, 1 / 2, 3]

    priors = []
    feature_maps = list(feature_map_dimensions.keys())

    for f, feature in enumerate(feature_maps):
        feature_dimension = feature_map_dimensions[feature]
        for i in range(feature_dimension):
            for j in range(feature_dimension):
                cx = (i + 0.5) / feature_dimension
                cy = (j + 0.5) / feature_dimension
                for ratio in aspect_ratios:
                    w = prior_scales[feature] * math.sqrt(ratio)
                    h = prior_scales[feature] / math.sqrt(ratio)
                    priors.append([cx, cy, w, h])

    priors = torch.FloatTensor(priors).clamp(0, 1)
    return priors


class SSD7(nn.Module):
    def __init__(self, num_classes):
        super(SSD7, self).__init__()
        self.backbone = Backbone()
        self.prediction = PredictionConv(num_classes)
        self.priors = get_priors()

    def forward(self, x):
        out4, out5, out6, out7 = self.backbone(x)
        locs, confs = self.prediction(out4, out5, out6, out7)

        output = (
            locs, torch.softmax(confs, dim=2), self.priors
        )
        return output


if __name__ == '__main__':
    img = torch.ones((2, 3, 250, 250))
    model = SSD7(num_classes=4)
    o = model(img)