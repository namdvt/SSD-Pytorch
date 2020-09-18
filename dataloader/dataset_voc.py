from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import random_split
from dataloader.data_transformation import DataTransformation

torch.manual_seed(0)


class VOCDataset(Dataset):
    def __init__( self, root, augment=False ):
        super().__init__()
        self.folder = root
        voc_labels = (
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        self.label_map = {k: v for v, k in enumerate(voc_labels)}
        self.indexes = open(root + '/ImageSets/Main/trainval.txt').read().splitlines()
        self.augment = augment

    def __getitem__( self, index ):
        image_name = self.indexes[index]
        image = Image.open(self.folder + '/JPEGImages/' + image_name + '.jpg').convert('RGB')

        labels = list()
        bboxes = list()
        difficulties = list()

        xml = ET.parse(self.folder + '/Annotations/' + image_name + '.xml')
        root = xml.getroot()
        for obj in root.iter('object'):
            label = obj.find('name').text

            bbox = obj.find('bndbox')
            xmax = int(bbox.find('xmax').text)
            xmin = int(bbox.find('xmin').text)
            ymax = int(bbox.find('ymax').text)
            ymin = int(bbox.find('ymin').text)

            # difficult = int(obj.find('difficult').text)

            labels.append(self.label_map.get(label))
            bboxes.append([xmin, ymin, xmax, ymax])
            # difficulties.append(difficult)

        bboxes = torch.FloatTensor(bboxes) / torch.FloatTensor([image.width, image.height, image.width, image.height])
        # difficulties = torch.Tensor(difficulties)

        # show2(image, labels, bboxes)
        return image, labels, bboxes

    def __len__(self):
        return len(self.indexes)


def collate_fn(batch):
    images = list()
    labels = list()
    bboxes = list()
    # difficulties = list()

    for b in batch:
        images.append(b[0])
        labels.append(b[1])
        bboxes.append(b[2])
        # difficulties.append(b[3])

    images = torch.stack(images)
    return images, labels, bboxes


def split(dataset):
    num_train = int(len(dataset) * 0.8)
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    return train_dataset, val_dataset


def get_loader(batch_size):
    # dataset_2012 = VOCDataset(root='data/VOC2012/35388_47853_bundle_archive/VOC2012')
    dataset_2007 = VOCDataset(root='data/VOCdevkit/VOC2007')

    # train_2012, val_2012 = split(dataset_2012)
    train_2007, val_2007 = split(dataset_2007)

    # train_2012 = DataTransformation(train_2012, augment=True)
    # val_2012 = DataTransformation(val_2012)
    train_2007 = DataTransformation(train_2007, augment=True)
    val_2007 = DataTransformation(val_2007)

    # train_loader = DataLoader(dataset=ConcatDataset([train_2007, train_2012]),
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           collate_fn=collate_fn,
    #                           drop_last=True,
    #                           num_workers=10,
    #                           pin_memory=True)
    # val_loader = DataLoader(dataset=ConcatDataset([val_2007, val_2012]),
    #                         batch_size=batch_size,
    #                         shuffle=True,
    #                         collate_fn=collate_fn,
    #                         drop_last=True,
    #                         num_workers=10,
    #                         pin_memory=True)

    train_loader = DataLoader(dataset=train_2007,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              drop_last=True,
                              # num_workers=10,
                              pin_memory=True)
    val_loader = DataLoader(dataset=val_2007,
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
