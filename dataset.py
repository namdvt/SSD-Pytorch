from PIL import Image

from helper import show, show2
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as F
import torch


class AfricanWildlifeDataset(Dataset):
    def __init__(self, root):
        super(AfricanWildlifeDataset, self).__init__()
        self.root = root
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
            bbox = [float(info[1]), float(info[2]), float(info[3]), float(info[4])]

            labels.append(label)
            bboxes.append(bbox)

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

    labels = torch.LongTensor(labels)

    return image, labels, bboxes


def get_loader(root, batch_size):
    dataset = AfricanWildlifeDataset(root)

    num_train = int(len(dataset) * 0.9)
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True, drop_last=True)

    return train_loader, val_loader


if __name__ == '__main__':
    train, val = get_loader('data/African_Wildlife/train', batch_size=2)
    for image, labels, bboxes in train:
        print()
    print()