from PIL import Image

from helper import show2
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as F
import torch
import numpy as np
from dataloader.data_transformation import DataTransformation

torch.manual_seed(0)

class AfricanWildlifeDataset(Dataset):
    def __init__(self, root):
        super(AfricanWildlifeDataset, self).__init__()
        self.root = root
        # self.image_list = open('../' + root + '/annotations.txt').read().splitlines()
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
            cx, cy, w, h = float(info[1]), float(info[2]), float(info[3]), float(info[4])
            bbox = np.asarray([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
            labels.append(label)
            bboxes.append(bbox)

        return image, labels, torch.tensor(bboxes, dtype=torch.float)


def collate_fn(batch):
    images = list()
    labels = list()
    bboxes = list()

    for b in batch:
        images.append(b[0])
        labels.append(b[1])
        bboxes.append(b[2])

    images = torch.stack(images)

    return images, labels, bboxes


def split(dataset):
    num_train = int(len(dataset) * 0.8)
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    return train_dataset, val_dataset


def get_loader(batch_size):
    dataset = AfricanWildlifeDataset(root='data/AfricanWildlife/')

    train, val = split(dataset)

    train = DataTransformation(train, augment=True)
    val = DataTransformation(val)

    train_loader = DataLoader(dataset=train,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              drop_last=True,
                              # num_workers=10,
                              pin_memory=True)
    val_loader = DataLoader(dataset=val,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn,
                            drop_last=True,
                            # num_workers=10,
                            pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    train, val = get_loader(batch_size=2)
    for image, labels, bboxes in train:
        print()
    print()
