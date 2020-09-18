import torch
import torch.nn as nn
import math

from torchvision.models import MobileNetV2, mobilenet_v2
from torchvision.models.mobilenet import InvertedResidual

from loss import cxcy_to_xy, gcxgcy_to_cxcy, cal_IoU


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
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
        num_priors = {'feat38': 4, 'feat19': 6, 'feat10': 6, 'feat5': 6, 'feat3': 4, 'feat1': 4}
        self.conv1 = PredictionLayerConv(32, num_priors['feat38'], num_classes)
        self.conv2 = PredictionLayerConv(96, num_priors['feat19'], num_classes)
        self.conv3 = PredictionLayerConv(1280, num_priors['feat10'], num_classes)
        self.conv4 = PredictionLayerConv(512, num_priors['feat5'], num_classes)
        self.conv5 = PredictionLayerConv(256, num_priors['feat3'], num_classes)
        self.conv6 = PredictionLayerConv(64, num_priors['feat1'], num_classes)

        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.modules():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)

    def forward(self, out38, out19, out10, out5, out3, out1):
        loc38, pred38 = self.conv1(out38)
        loc19, pred19 = self.conv2(out19)
        loc10, pred10 = self.conv3(out10)
        loc5, pred5 = self.conv4(out5)
        loc3, pred3 = self.conv5(out3)
        loc1, pred1 = self.conv6(out1)

        locs = torch.cat([loc38, loc19, loc10, loc5, loc3, loc1], dim=1)
        preds = torch.cat([pred38, pred19, pred10, pred5, pred3, pred1], dim=1)

        return locs, preds


class LayerActivation:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


class MobileNet(nn.Module):
    def __init__(self, device, freeze=True):
        super(MobileNet, self).__init__()
        self.device = device
        self.mobilenet = mobilenet_v2(pretrained=True, progress=True).to(device)
        if freeze:
            for m in self.mobilenet.parameters():
                m.requires_grad = False

        self.layer6 = LayerActivation(self.mobilenet.features, 6)
        self.layer13 = LayerActivation(self.mobilenet.features, 13)
        self.layer18 = LayerActivation(self.mobilenet.features, 18)

    def forward(self, x):
        self.mobilenet(x)
        out38 = self.layer6.features.to(self.device)
        out19 = self.layer13.features.to(self.device)
        out10 = self.layer18.features.to(self.device)

        return out38, out19, out10


class AuxiliaryConv(nn.Module):
    def __init__(self):
        super(AuxiliaryConv, self).__init__()
        self.conv1 = InvertedResidual(1280, 512, stride=2, expand_ratio=0.2)
        self.conv2 = InvertedResidual(512, 256, stride=2, expand_ratio=0.25)
        self.conv3 = InvertedResidual(256, 256, stride=2, expand_ratio=0.5)
        self.conv4 = InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.modules():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        out5 = self.conv1(x)
        out3 = self.conv2(out5)
        out2 = self.conv3(out3)
        out1 = self.conv4(out2)

        return out5, out3, out1


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
    def __init__(self, num_classes, device, freeze=False):
        super(SSD_MobileNet, self).__init__()
        self.num_classes = num_classes
        self.mobilenet = MobileNet(device=device, freeze=freeze).to(device)
        self.auxiliary = AuxiliaryConv().to(device)
        self.prediction = PredictionConv(num_classes).to(device)
        self.priors_cxcy = get_priors().to(device)

    def forward(self, x):
        out38, out19, out10 = self.mobilenet(x)
        out5, out3, out1 = self.auxiliary(out10)
        locs, confs = self.prediction(out38, out19, out10, out5, out3, out1)

        return locs, confs


# if __name__ == '__main__':
#     model = SSD_MobileNet(num_classes=21, device='cpu')
#     # model = SSD_MobileNet()
#
#     x = torch.zeros((2, 3, 256, 256))
#     out = model(x)
#     print()