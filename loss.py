import torch
import torch.nn as nn


def cal_IoU(boxes1, boxes2):
    intersection = cal_intersection(boxes1, boxes2)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    return intersection / union


def cal_intersection(boxes1, boxes2):
    lower_bounds = torch.max(boxes1[:, :2].unsqueeze(1), boxes2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def xy_to_cxcywh(xy):
    cxcy = (xy[:, 2:] + xy[:, :2]) / 2
    wh = xy[:, 2:] - xy[:, :2]
    return torch.cat([cxcy, wh], dim=1)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


class MultiBoxLoss(nn.Module):
    def __init__(self, priors, device, num_classes):
        super(MultiBoxLoss, self).__init__()
        self.device = device
        self.priors_cxcy = priors
        self.priors = cxcy_to_xy(priors)
        self.num_priors = self.priors.shape[0]
        self.smooth_l1 = nn.SmoothL1Loss()
        self.negative_ratio = 3
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes

    def forward(self, pred_locs, pred_scores, bboxes, labels):
        batch_size = pred_locs.shape[0]
        true_classes = torch.zeros((batch_size, self.num_priors), dtype=torch.long).to(self.device)
        true_locs = torch.zeros((batch_size, self.num_priors, 4), dtype=torch.float).to(self.device)

        for i in range(0, batch_size):
            # n_objects = bboxes[i].size(0)
            n_objects = len(bboxes[i])

            iou = cal_IoU(self.priors, bboxes[i])
            iou_for_each_prior, object_for_each_prior = iou.max(dim=1)
            _, prior_for_each_object = iou.max(dim=0)

            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects))

            iou_for_each_prior[prior_for_each_object] = 1
            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[iou_for_each_prior < 0.5] = 0
            true_classes[i] = label_for_each_prior
            # true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcywh(bboxes[i][object_for_each_prior]), self.priors_cxcy)
            true_locs[i] = cxcy_to_gcxgcy(bboxes[i][object_for_each_prior], self.priors_cxcy)
            print()

        positive_priors = true_classes != 0
        # Localization loss
        loc_loss = self.smooth_l1(pred_locs[positive_priors].to(self.device),
                                  true_locs[positive_priors].to(self.device))

        # confidence loss
        num_positives = positive_priors.sum(dim=1)
        num_hard_negatives = self.negative_ratio * num_positives

        # positive loss
        conf_loss_all = self.cross_entropy(pred_scores.view(-1, self.num_classes), true_classes.view(-1)).view(batch_size, -1)
        conf_loss_positive = conf_loss_all[positive_priors]

        # hard negative loss
        conf_loss_negative = conf_loss_all.clone()
        conf_loss_negative[positive_priors] = 0
        conf_loss_negative, _ = conf_loss_negative.sort(dim=1, descending=True)

        hardness_ranks = torch.LongTensor(range(8732)).unsqueeze(0).expand_as(conf_loss_negative).to(self.device)
        hard_negatives = hardness_ranks < num_hard_negatives.unsqueeze(1)
        conf_loss_hard_negatives = conf_loss_negative[hard_negatives]

        conf_loss = (conf_loss_hard_negatives.sum() + conf_loss_positive.sum()) / num_positives.sum().float()

        return conf_loss + loc_loss
