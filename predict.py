import dataset as dset
from helper import show2
from loss import gcxgcy_to_cxcy, cal_IoU, cxcy_to_xy
from model2 import SSD7
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

model = SSD7(num_classes=5).to(device)
model.load_state_dict(torch.load('output/weight.pth', map_location=device))
model.eval()


# def detect(predicted_locs, predicted_scores, min_score=0.2, max_overlap=0.5, top_k=5):
#     print()
#     predicted_scores = torch.softmax(predicted_scores, dim=1)
#     max_scores, best_label = predicted_scores.max(dim=1)
#     decoded_locs = gcxgcy_to_cxcy(predicted_locs, model.priors)
#
#     for c in range(1, 5):
#         class_score = predicted_scores[:, c]
#         score_above_min = class_score > min_score
#         if score_above_min.sum() == 0:
#             continue
#
#         # filter bbs have high score
#         class_score = class_score[score_above_min]
#         class_locs = decoded_locs[score_above_min]
#
#         # sort
#         class_score, idx = class_score.sort(descending=True)
#         class_locs = class_locs[idx]
#
#         show2(image, class_score, class_locs)
#
#         overlap = cal_IoU(class_locs, class_locs)
#         print()


def detect_objects(predicted_locs, predicted_scores, min_score=0.2, max_overlap=0.5, top_k=5):
    """
    Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
    For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
    :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
    :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
    :param min_score: minimum threshold for a box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :return: detections (boxes, labels, and scores), lists of length batch_size
    """
    batch_size = predicted_locs.size(0)
    n_priors = model.priors.size(0)
    predicted_scores = torch.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted boxes to
        decoded_locs = cxcy_to_xy(
            gcxgcy_to_cxcy(predicted_locs[i], model.priors))  # (8732, 4), these are fractional pt. coordinates

        # Lists to store boxes and scores for this image
        image_boxes = list()
        image_labels = list()
        image_scores = list()

        max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

        # Check for each class
        for c in range(1, 5):
            # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = predicted_scores[i][:, c]  # (8732)
            score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
            class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

            # Sort predicted boxes and scores by scores
            class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
            class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

            # Find the overlap between predicted boxes
            overlap = cal_IoU(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

            # Non-Maximum Suppression (NMS)

            # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
            # 1 implies suppress, 0 implies don't suppress
            suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box is already marked for suppression
                if suppress[box] == 1:
                    continue

                # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                # Find such boxes and update suppress indices
                suppress = torch.max(suppress, overlap[box] > max_overlap)
                # The max operation retains previously suppressed boxes, like an 'OR' operation

                # Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = 0

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[1 - suppress])
            image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
            image_scores.append(class_scores[1 - suppress])

        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            image_labels.append(torch.LongTensor([0]).to(device))
            image_scores.append(torch.FloatTensor([0.]).to(device))

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        # Keep only the top k objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]  # (top_k)
            image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
            image_labels = image_labels[sort_ind][:top_k]  # (top_k)

        # Append to lists that store predicted boxes and scores for all images
        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


if __name__ == '__main__':
    image = Image.open('data/African_Wildlife/train/buffalo/002.jpg').convert('RGB')
    image = F.resize(image, size=(250, 250))
    image = F.to_tensor(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        predicted_locs, predicted_scores = model(image)

    detect_objects(predicted_locs, predicted_scores)

    # show(image, det_labels[0], det_boxes[0])