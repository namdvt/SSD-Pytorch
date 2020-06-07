import matplotlib.pyplot as plt
import matplotlib.patches as patches

coco_labels = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
               8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: '', 12: 'stop sign', 13: 'parking meter',
               14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant',
               22: 'bear', 23: 'zebra', 24: 'giraffe', 25: '', 26: 'backpack', 27: 'umbrella', 28: '', 29: '',
               30: 'handbag', 31: 'tie', 32: 'suitcase', 33: 'frisbee', 34: 'skis', 35: 'snowboard', 36: 'sports ball',
               37: 'kite', 38: 'baseball bat', 39: 'baseball glove', 40: 'skateboard', 41: 'surfboard',
               42: 'tennis racket', 43: 'bottle', 44: '', 45: 'wine glass', 46: 'cup', 47: 'fork', 48: 'knife',
               49: 'spoon', 50: 'bowl', 51: 'banana', 52: 'apple', 53: 'sandwich', 54: 'orange', 55: 'broccoli',
               56: 'carrot', 57: 'hot dog', 58: 'pizza', 59: 'donut', 60: 'cake', 61: 'chair', 62: 'couch',
               63: 'potted plant', 64: 'bed', 65: '', 66: 'dining table', 67: '', 68: '', 69: 'toilet', 70: '',
               71: 'tv', 72: 'laptop', 73: 'mouse', 74: 'remote', 75: 'keyboard', 76: 'cell phone', 77: 'microwave',
               78: 'oven', 79: 'toaster', 80: 'sink', 81: 'refrigerator', 82: '', 83: 'book', 84: 'clock', 85: 'vase',
               86: 'scissors', 87: 'teddy bear', 88: 'hair drier', 89: 'toothbrush'}


def show(image, labels, bboxes):
    bboxes = bboxes * 300
    plt.imshow(image.squeeze().permute(1, 2, 0))
    figure = plt.gca()
    for label, bbox in zip(labels, bboxes):
        rectangle = patches.Rectangle(xy=(bbox[0], bbox[1]),
                                      width=bbox[2],
                                      height=bbox[3],
                                      linewidth=1, edgecolor='r', facecolor='none')
        figure.add_patch(rectangle)
        plt.text(bbox[0] + 5, bbox[1], coco_labels[label],
                 fontsize=9, bbox=dict(color='red', alpha=0, fill=True))

    plt.savefig('result.png')
    plt.close()
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