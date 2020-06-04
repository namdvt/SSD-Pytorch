from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from helper import show

from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import tifffile
import torchvision.transforms.functional as F
import cv2
import torch.nn.functional as TF
import torch
import random


class CocoDataset(Dataset):
    def __init__(self, folder, anns, size):
        super(CocoDataset, self).__init__()
        self.data_folder = folder
        self.size = size

        self.coco = COCO(anns)
        self.img_list = list(self.coco.imgs.keys())
        self.category_ids = self.coco.getCatIds()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_id = self.img_list[index]
        image = Image.open(self.data_folder + self.coco.imgs.get(image_id).get('file_name'))

        if image.layers != 3:
            image = image.expand(3, image.width, image.height)

        labels = list()
        bboxes = list()

        annotations_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids, iscrowd=None)
        annotations = self.coco.loadAnns(annotations_ids)

        for ann in annotations:
            labels.append(ann['category_id'] - 1)
            bboxes.append(torch.tensor(ann['bbox']).float())

        image, labels, bboxes = transform(image, labels, torch.stack(bboxes))
        show(image, labels, bboxes)
        return image, labels, bboxes


def collate(batch):
    images = list()
    labels = list()
    bboxes = list()

    for b in batch:
        images.append(b[0])
        labels.append(b[1])
        bboxes.append(b[2])

    images = torch.stack(images)
    return images, labels, bboxes


def transform(image, labels, bboxes):
    ratio = 300 / torch.FloatTensor([image.width, image.height, image.width, image.height])
    bboxes = torch.FloatTensor(bboxes) * ratio / 300

    image = F.resize(image, size=(300, 300))
    image = F.to_tensor(image)

    return image, labels, bboxes


if __name__ == '__main__':
    anns_file = 'data/annotations/instances_val2017.json'
    data_folder = 'data/val2017/'
    size = 300

    coco_dataset = CocoDataset(anns=anns_file, folder=data_folder, size=size)
    coco_dataloader = DataLoader(coco_dataset, batch_size=2, collate_fn=collate, shuffle=True, drop_last=True)
    for image, labels, bboxes in coco_dataloader:
        print()

