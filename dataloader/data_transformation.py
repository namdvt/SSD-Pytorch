from torch.utils.data import Dataset, DataLoader, ConcatDataset
import xml.etree.ElementTree as ET
from PIL import Image, ImageFilter
import torchvision.transforms.functional as F
import torch
from torch.utils.data import random_split
from helper import show2
import random
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transform

from loss import cal_IoU


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [F.adjust_brightness,
                   F.adjust_contrast,
                   F.adjust_saturation,
                   F.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.
    Note that some objects may be cut out entirely.
    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = cal_IoU(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def zoom_out(image, bboxes, filler):
    w, h = image.size
    scale_ratio = (random.randint(4, 10) / 10, random.randint(4, 10) / 10)
    # scale_ratio = (.2, .2)
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale_ratio[0]
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale_ratio[1]
    new_size = (int(scale_ratio[0] * w), int(scale_ratio[1] * h))
    resized_image = F.resize(image, (new_size[1], new_size[0]))
    image = torch.ones((3, h, w), dtype=torch.float) * torch.FloatTensor(filler).unsqueeze(1).unsqueeze(1)
    image = F.to_pil_image(image)

    align = (random.randint(0, abs(w - new_size[0])), random.randint(0, abs(h - new_size[1])))
    image.paste(resized_image, align)
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + align[0] / w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + align[1] / h
    bboxes = torch.clamp(bboxes, min=0, max=1)

    return image, bboxes


def horizontal_flip(image, bboxes):
    image = F.hflip(image)
    bboxes = torch.abs(torch.tensor([1, 0, 1, 0]).float() - bboxes)
    bboxes[:, [0, 2]] = bboxes[:, [2, 0]]
    return image, bboxes


class DataTransformation(Dataset):
    def __init__(self, dataset, augment=False):
        self.dataset = dataset
        self.augment = augment

    def __getitem__(self, idx):
        image, labels, bboxes = self.dataset[idx]

        if self.augment:
            # adjust brighness, contrast, saturation
            image = photometric_distort(image)

            # zoom out
            image, bboxes = zoom_out(image, bboxes, filler=[0.485, 0.456, 0.406])

            # horizontal flip
            if random.choice([0, 1]):
                image, bboxes = horizontal_flip(image, bboxes)

        # resize and return
        image = F.resize(image, size=(300, 300))
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        labels = torch.LongTensor(labels)

        return image, labels, bboxes

    def __len__(self):
        return len(self.dataset)