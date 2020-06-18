import torch
import torch.optim as optim
from helper import write_log, write_figure
import numpy as np
from dataset import get_loader
from model import SSD
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from loss import MultiBoxLoss
import math


def fit(epoch, model, optimizer, criterion, device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0

    for image, labels, bboxes in tqdm(data_loader):
        image = image.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            pred_locs, pred_scores = model(image)
        else:
            with torch.no_grad():
                pred_locs, pred_scores = model(image)

        loss = criterion(pred_locs, pred_scores, bboxes, labels)
        if math.isnan(loss):
            criterion(pred_locs, pred_scores, bboxes, labels)
        print(loss.item())
        running_loss += loss.item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader.dataset)
    print('[%d][%s] loss: %.4f' % (epoch, phase, epoch_loss))
    return epoch_loss


def train():
    print('start training ...........')
    batch_size = 2
    num_epochs = 200
    learning_rate = 0.1

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = SSD(num_classes=5, device=device)
    # model.load_state_dict(torch.load('output_3/weight.pth', map_location=device))

    train_loader, val_loader = get_loader('data/African_Wildlife/train', batch_size=batch_size)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = MultiBoxLoss(priors=model.priors, device=device, num_classes=5)

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_epoch_loss = fit(epoch, model, optimizer, criterion, device, train_loader, phase='training')
        val_epoch_loss = fit(epoch, model, optimizer, criterion, device, val_loader, phase='validation')
        print('-----------------------------------------')

        if epoch == 0 or val_epoch_loss <= np.min(val_losses):
            torch.save(model.state_dict(), 'output/weight.pth')

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        write_figure('output', train_losses, val_losses)
        write_log('output', epoch, train_epoch_loss, val_epoch_loss)

        scheduler.step(val_epoch_loss)


if __name__ == "__main__":
    train()
