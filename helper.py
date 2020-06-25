import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

African_wildlife_labels = {0: 'buffalo', 1: 'elephant', 2: 'rhino', 3: 'zebra'}


def show(image, labels, bboxes):
    bboxes = torch.tensor(bboxes) * torch.FloatTensor([image.width, image.height, image.width, image.height])
    plt.imshow(image)
    figure = plt.gca()
    for label, bbox in zip(labels, bboxes):
        rectangle = patches.Rectangle(xy=(bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2),
                                      width=bbox[2],
                                      height=bbox[3],
                                      linewidth=1, edgecolor='r', facecolor='none')
        figure.add_patch(rectangle)
        plt.text(bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, African_wildlife_labels[label],
                 fontsize=9, bbox=dict(color='red', alpha=0, fill=True))

    plt.savefig('result.png')
    plt.show()
    print()


def show2(image, labels, bboxes):
    bboxes = bboxes * 250
    plt.imshow(image.squeeze().permute(1, 2, 0))
    figure = plt.gca()
    for label, bbox in zip(labels, bboxes):
        rectangle = patches.Rectangle(xy=(bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2),
                                      width=bbox[2],
                                      height=bbox[3],
                                      linewidth=1, edgecolor='r', facecolor='none')
        figure.add_patch(rectangle)
        plt.text(bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, African_wildlife_labels[label.item()],
                 fontsize=9, bbox=dict(color='red', alpha=0, fill=True))

    plt.show()
    print()


def write_figure(location, train_losses, val_losses):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.savefig(location + '/loss.png')
    plt.close('all')


def write_log(location, epoch, train_loss, val_loss):
    if epoch == 0:
        f = open(location + '/log.txt', 'w+')
        f.write('epoch\t\ttrain_loss\t\tval_loss\n')
    else:
        f = open(location + '/log.txt', 'a+')

    f.write(str(epoch) + '\t' + str(train_loss) + '\t' + str(val_loss) + '\n')

    f.close()
