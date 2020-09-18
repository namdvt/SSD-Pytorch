import torch
import torch.nn as nn
import math

from loss import cxcy_to_xy, gcxgcy_to_cxcy, cal_IoU


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
        self.localization_conv = nn.Conv2d(in_channels, dimensions * 4, kernel_size=3, padding=1)
        self.prediction_conv = nn.Conv2d(in_channels, dimensions * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        bs = x.shape[0]
        localization = self.localization_conv(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
        prediction = self.prediction_conv(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)

        return localization, prediction


# -----------------------MOBILE NET------------------------------------------

class PredictionConv(nn.Module):
    def __init__(self, num_classes):
        super(PredictionConv, self).__init__()
        num_priors = {'conv4': 4, 'conv5': 6, 'conv6': 6, 'conv7': 6, 'conv8': 4, 'conv9': 4}
        self.conv4 = PredictionLayerConv(256, num_priors['conv4'], num_classes)
        self.conv5 = PredictionLayerConv(512, num_priors['conv5'], num_classes)
        self.conv6 = PredictionLayerConv(1024, num_priors['conv6'], num_classes)
        self.conv7 = PredictionLayerConv(2048, num_priors['conv7'], num_classes)
        self.conv8 = PredictionLayerConv(2048, num_priors['conv8'], num_classes)
        self.conv9 = PredictionLayerConv(2048, num_priors['conv9'], num_classes)

    def forward(self, out4, out5, out6, out7, out8, out9):
        loc4, pred4 = self.conv4(out4)
        loc5, pred5 = self.conv5(out5)
        loc6, pred6 = self.conv6(out6)
        loc7, pred7 = self.conv7(out7)
        loc8, pred8 = self.conv8(out8)
        loc9, pred9 = self.conv9(out9)

        locs = torch.cat([loc4, loc5, loc6, loc7, loc8, loc9], dim=1)
        preds = torch.cat([pred4, pred5, pred6, pred7, pred8, pred9], dim=1)

        return locs, preds


class DepthWiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthWiseSeparableConv2d, self).__init__()
        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)
        )
        self.point_wise_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.depth_wise_conv(x)
        x = self.point_wise_conv(x)
        return x


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 = DepthWiseSeparableConv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = DepthWiseSeparableConv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = DepthWiseSeparableConv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = DepthWiseSeparableConv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv6 = DepthWiseSeparableConv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = DepthWiseSeparableConv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.conv8 = nn.Sequential(
            DepthWiseSeparableConv2d(512, 512, kernel_size=3, padding=1),
            DepthWiseSeparableConv2d(512, 512, kernel_size=3, padding=1),
            DepthWiseSeparableConv2d(512, 512, kernel_size=3, padding=1),
            DepthWiseSeparableConv2d(512, 512, kernel_size=3, padding=1),
            DepthWiseSeparableConv2d(512, 512, kernel_size=3, padding=1)
        )
        self.conv9 = DepthWiseSeparableConv2d(512, 1024, kernel_size=3, padding=1, stride=2)
        self.conv10 = DepthWiseSeparableConv2d(1024, 1024, kernel_size=3, padding=1)
        self.conv11 = DepthWiseSeparableConv2d(1024, 1024, kernel_size=3, padding=1, stride=2)
        self.conv12 = DepthWiseSeparableConv2d(1024, 2048, kernel_size=3, padding=1)
        self.conv13 = DepthWiseSeparableConv2d(2048, 2048, kernel_size=3, padding=1, stride=2)
        self.conv14 = nn.Sequential(
            DepthWiseSeparableConv2d(2048, 2048, kernel_size=3, padding=1, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)    # 56
        x = self.conv5(x)
        out6 = self.conv6(x)    # 28
        x = self.conv7(out6)
        out8 = self.conv8(x)
        x = self.conv9(out8)
        out10 = self.conv10(x)
        x = self.conv11(out10)
        out12 = self.conv12(x)
        out13 = self.conv13(out12)
        out14 = self.conv14(out13)

        return out6, out8, out10, out12, out13, out14


def get_priors():
    fmap_dims = {'conv4_3': 38,
                 'conv7': 19,
                 'conv8_2': 10,
                 'conv9_2': 5,
                 'conv10_2': 3,
                 'conv11_2': 1}

    obj_scales = {'conv4_3': 0.1,
                  'conv7': 0.2,
                  'conv8_2': 0.375,
                  'conv9_2': 0.55,
                  'conv10_2': 0.725,
                  'conv11_2': 0.9}

    aspect_ratios = {'conv4_3': [1., 2., 0.5],
                     'conv7': [1., 2., 3., 0.5, .333],
                     'conv8_2': [1., 2., 3., 0.5, .333],
                     'conv9_2': [1., 2., 3., 0.5, .333],
                     'conv10_2': [1., 2., 0.5],
                     'conv11_2': [1., 2., 0.5]}

    fmaps = list(fmap_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append(
                        [cx, cy, obj_scales[fmap] * math.sqrt(ratio), obj_scales[fmap] / math.sqrt(ratio)])
                    if ratio == 1.:
                        try:
                            additional_scale = math.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes)
    prior_boxes.clamp_(0, 1)

    return prior_boxes


# -----------------------MOBILE NET------------------------------------------

class SSD_MobileNet(nn.Module):
    def __init__(self, num_classes, device):
        super(SSD_MobileNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = MobileNet().to(device)
        self.prediction = PredictionConv(num_classes).to(device)
        self.priors_cxcy = get_priors().to(device)
        self._initialize_weights()

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

    def forward(self, x):
        out4, out5, out6, out7, out8, out9 = self.backbone(x)
        locs, confs = self.prediction(out4, out5, out6, out7, out8, out9)

        return locs, confs


# if __name__ == '__main__':
#     model = SSD_MobileNet(num_classes=21, device='cpu')
#     # model = MobileNet()
#
#     x = torch.zeros((2, 3, 300, 300))
#     out = model(x)
#     print()