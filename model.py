import torch
import torch.nn as nn
import math
import torchvision.models
from torch.utils import model_zoo
import torch.nn.functional as F

from loss import cxcy_to_xy, cal_IoU, gcxgcy_to_cxcy

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
        x = F.dropout2d(x, 0.3)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, freezed=True):
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

        if freezed:
            for param in self.features.parameters():
                param.requires_grad = False

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
        self.localization_conv = nn.Conv2d(in_channels, dimensions * 4, kernel_size=3, padding=1)
        self.prediction_conv = nn.Conv2d(in_channels, dimensions * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        bs = x.shape[0]
        localization = self.localization_conv(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
        prediction = self.prediction_conv(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)

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


class SSD(nn.Module):
    def __init__(self, num_classes, device):
        super(SSD, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.vgg = vgg16(pretrained=True, freezed=True).to(device)
        self.auxiliary_conv = AuxiliaryConv().to(device)
        self.prediction_conv = PredictionConv(num_classes).to(device)
        self.priors_cxcy = get_priors().to(device)

    def forward(self, x):
        out4_3, out5_3 = self.vgg(x, self.device)
        out7, out8, out9, out10, out11 = self.auxiliary_conv(out5_3)
        locs, preds = self.prediction_conv(out4_3, out7, out8, out9, out10, out11)

        return locs, preds

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.num_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = cal_IoU(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(self.device)  # (n_qualified)

                suppress = torch.zeros((n_above_min_score)).bool().to(self.device)
                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    # suppress = torch.max(suppress, overlap[box] > max_overlap)
                    suppress = suppress | (overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(
                    torch.LongTensor(
                        (~suppress).sum().item() * [c]).to(self.device)
                )
                image_scores.append(class_scores[~suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
                image_labels.append(torch.LongTensor([0]).to(self.device))
                image_scores.append(torch.FloatTensor([0.]).to(self.device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


if __name__ == '__main__':
    img = torch.ones((2, 3, 300, 300))
    model = SSD(num_classes=4, device='cpu')
    o = model(img)
