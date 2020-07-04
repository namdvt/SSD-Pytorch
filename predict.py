from torchvision import transforms
from PIL import Image
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from helper import African_wildlife_labels

from model import SSD

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    # Transform
    image = F.resize(original_image, size=(300, 300))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.to(device)

    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    det_boxes = det_boxes[0].to('cpu')

    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # if det_labels == ['background']:
    #     return original_image

    # Annotate
    plt.imshow(original_image)
    figure = plt.gca()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        bbox = det_boxes[i].tolist()
        rectangle = patches.Rectangle(xy=(bbox[0], bbox[1]),
                                      width=bbox[2] - bbox[0],
                                      height=bbox[3] - bbox[1],
                                      linewidth=1, edgecolor='r', facecolor='none')
        figure.add_patch(rectangle)
        plt.text(bbox[0], bbox[1], African_wildlife_labels[det_labels[0][i].item()],
                 fontsize=20, bbox=dict(color='red', alpha=0, fill=True))
    plt.show()


if __name__ == '__main__':
    # Load model checkpoint
    device = torch.device("cpu")
    model = SSD(num_classes=21, device=device)
    # model.load_state_dict(torch.load('output/weight.pth', map_location=device))
    model = model.to(device)
    model.eval()

    img_path = 'data/African_Wildlife/test/buffalo/001.jpg'
    original_image = Image.open(img_path)

    detect(original_image, min_score=0.5, max_overlap=0.5, top_k=3)