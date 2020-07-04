from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms.functional as F
import torch
from torch.utils.data import random_split


class VOCDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.folder = root
        voc_labels = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        self.label_map = {k: v for v, k in enumerate(voc_labels)}
        self.indexes = open(root + '/ImageSets/Main/trainval_small.txt').read().splitlines()

    def __getitem__(self, index):
        image_name = self.indexes[index]
        image = Image.open(self.folder + '/JPEGImages/' + image_name + '.jpg').convert('RGB')

        labels = list()
        bboxes = list()

        xml = ET.parse(self.folder + '/Annotations/' + image_name + '.xml')
        root = xml.getroot()
        for obj in root.iter('object'):
            label = obj.find('name').text

            bbox = obj.find('bndbox')
            xmax = int(bbox.find('xmax').text)
            xmin = int(bbox.find('xmin').text)
            ymax = int(bbox.find('ymax').text)
            ymin = int(bbox.find('ymin').text)

            labels.append(self.label_map.get(label))
            bboxes.append([xmin, ymin, xmax, ymax])

        image, labels, bboxes = transform(image, labels, bboxes)
        return image, labels, bboxes

    def __len__(self):
        return len(self.indexes)


# resize and toTensor
def transform(image, labels, bboxes):
    bboxes = torch.FloatTensor(bboxes) / torch.FloatTensor([image.width, image.height, image.width, image.height])

    image = F.resize(image, size=(300, 300))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    labels = torch.LongTensor(labels)
    return image, labels, bboxes


def collate_fn(batch):
    images = list()
    labels = list()
    bboxes = list()

    for b in batch:
        images.append(b[0])
        labels.append(b[1])
        bboxes.append(b[2])

    images = torch.stack(images)
    # bboxes = torch.stack(bboxes)
    # bboxes = bboxes.view(bboxes.shape[1], bboxes.shape[2])
    return images, labels, bboxes


def get_loader(root, batch_size):
    dataset = VOCDataset(root=root)

    num_train = int(len(dataset) * 0.8)
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(dataset=train_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_fn,
                             drop_last=True)
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              drop_last=True)
    return train_loader, val_loader


if __name__ == '__main__':
    train, val = get_loader('data/VOC2012', batch_size=2)
    for image, labels, bboxes in train:
        print()
    print()
