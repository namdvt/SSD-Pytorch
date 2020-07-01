import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class AfricanWildlifeDataset(Dataset):
    def __init__(self, root, augment=False):
        super(AfricanWildlifeDataset, self).__init__()
        self.root = root
        self.augment = augment
        self.image_list = open(root + '/annotations.txt').read().splitlines()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image, annotation = self.image_list[index].split('\t')
        image = Image.open(image)

        labels = list()
        bboxes = list()

        annotation = open(annotation, 'r')
        for line in annotation.readlines():
            info = line.split()
            label = int(line.split()[0]) + 1
            bbox = np.array([float(info[1]), float(info[2]), float(info[3]), float(info[4])])

            labels.append(label)
            bboxes.append(bbox)

        if self.augment:
            image, labels, bboxes = augment_data(image, labels, bboxes)
        image, labels, bboxes = transform(image, labels, bboxes)
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
    bboxes = torch.FloatTensor(bboxes)

    image = F.resize(image, size=(300, 300))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    labels = torch.LongTensor(labels)

    return image, labels, bboxes


def augment_data(image, labels, bboxes):
    # adjust brightness, contrast, saturation
    image = F.adjust_brightness(image, random.randint(5, 15) / 10)
    image = F.adjust_contrast(image, random.randint(5, 15) / 10)
    image = F.adjust_saturation(image, random.randint(0, 5))
    image = F.adjust_hue(image, random.randint(-5, 5) / 10)
    image = F.adjust_gamma(image, random.randint(3, 30) / 10)

    # horizontal flip
    if random.randint(0, 1):
        image = F.hflip(image)
        bboxes_copy = bboxes.copy()
        bboxes = list()
        for box in bboxes_copy:
            bboxes.append(abs([-1, 0, 0, 0] + box))

    return image, labels, bboxes


def get_loader(root, batch_size):
    train_dataset = AfricanWildlifeDataset(root + '/train', augment=True)
    val_dataset = AfricanWildlifeDataset(root + '/val', augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True, drop_last=True)

    return train_loader, val_loader


# if __name__ == '__main__':
#     train, val = get_loader('data/African_Wildlife/train', batch_size=2)
#     for image, labels, bboxes in train:
#         print()
#     print()
