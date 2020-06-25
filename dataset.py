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
        image = F.resize(image, size=(250, 250))
        image = F.to_tensor(image)

        target = list()

        annotation = open(annotation, 'r')
        for line in annotation.readlines():
            obj = line.split()
            obj = obj[1:] + [obj[0]]
            obj = [torch.tensor(float(i)) for i in obj]
            obj = torch.stack(obj)

            target.append(obj)
        target = torch.stack(target)

        return image, target


def collate(batch):
    images = list()
    targets = list()

    for b in batch:
        images.append(b[0])
        targets.append(b[1])

    images = torch.stack(images)

    return images, targets


def get_loader(root, batch_size):
    dataset = AfricanWildlifeDataset(root)

    num_train = int(len(dataset) * 0.9)
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True, drop_last=True)

    return train_loader, val_loader


if __name__ == '__main__':
    train, val = get_loader('data/African_Wildlife/train', batch_size=8)
    for images, targets in train:
        print()
    print()