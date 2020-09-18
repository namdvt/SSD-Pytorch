import torch
import torchvision.transforms.functional as F
from loss import cxcy_to_xy, gcxgcy_to_cxcy, cal_IoU
from models.ssd_mobilenet_v2 import SSD_MobileNet
from PIL import Image, ImageFont, ImageDraw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = {0: 'background', 1: ('buffalo', '#e6194b'), 2: ('elephant', '#3cb44b'),
          3: ('rhino', '#1035F2'),
          4: ('zebra', '#911eb4')}


def detect_objects( model, predicted_locs, predicted_scores, min_score, max_overlap, top_k ):
    batch_size = predicted_locs.size()[0]
    n_priors = model.priors_cxcy.size()[0]
    predicted_scores = torch.softmax(predicted_scores, dim=2)

    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):
        decoded_locs = cxcy_to_xy(
            gcxgcy_to_cxcy(predicted_locs[i], model.priors_cxcy))

        image_boxes = list()
        image_labels = list()
        image_scores = list()

        for c in range(1, model.num_classes):
            # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = predicted_scores[i][:, c]
            score_above_min_score = class_scores > min_score
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores = class_scores[score_above_min_score]
            class_decoded_locs = decoded_locs[score_above_min_score]

            # Sort predicted boxes and scores by scores
            class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
            class_decoded_locs = class_decoded_locs[sort_ind]

            # Find the overlap between predicted boxes
            overlap = cal_IoU(class_decoded_locs, class_decoded_locs)

            # Non-Maximum Suppression (NMS)
            # A torch.uint8 (byte)tensor to keep track of which predicted boxes to suppress
            # 1 implies suppress, 0 implies don't suppress
            suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box is already marked for suppression
                if suppress[box] == 1:
                    continue

                # Suppress boxes whose overlaps (with this box)are greater than maximum overlap
                # Find such boxes and update suppress indices
                condition = overlap[box] > max_overlap
                condition = condition.clone().detach().type(torch.uint8)
                suppress = torch.max(suppress, condition)

                # Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = 0

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[(1 - suppress).type(torch.bool)])
            image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]))
            image_scores.append(class_scores[(1 - suppress).type(torch.bool)])

        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]))
            image_labels.append(torch.LongTensor([0]))
            image_scores.append(torch.FloatTensor([0.]))

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


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    # Transform
    image = F.resize(original_image, size=(300, 300))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.to(device)

    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    det_boxes, det_labels, det_scores = detect_objects(model, predicted_locs, predicted_scores, min_score=min_score,
                                                       max_overlap=max_overlap, top_k=top_k)
    det_boxes = det_boxes[0].to('cpu')

    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        clazz = det_labels[0][i].item()
        if clazz == 0:
            continue

        bbox = det_boxes[i].tolist()
        name = labels[clazz][0]
        color = labels[clazz][1]

        # draw bbox
        box_location = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
        draw = ImageDraw.Draw(original_image)
        draw.rectangle(xy=box_location, outline=color, width=3)

        # write text
        font = ImageFont.truetype("Ubuntu-B.ttf", 15)
        text_size = font.getsize(labels[clazz][1].upper())
        text_location = [bbox[0] + 2., bbox[1] - text_size[1]]
        textbox_location = [bbox[0], bbox[1] - text_size[1], bbox[0] + text_size[0] + 12., bbox[1]]

        draw.rectangle(xy=textbox_location, fill=color)
        draw.text(xy=text_location, text=name.upper(), fill='white', font=font)

        original_image.save("output/4.jpg", "JPEG")


if __name__ == '__main__':
    model = SSD_MobileNet(num_classes=5, device=device)
    model.load_state_dict(torch.load('output/weight_wildlife.pth', map_location=device))
    model = model.to(device)
    model.eval()

    img_path = '/home/nam/Desktop/4.jpg'
    original_image = Image.open(img_path)

    w = 800
    h = int(w * original_image.height / original_image.width)
    original_image = original_image.resize((w, h))

    detect(original_image, min_score=0.3, max_overlap=0.4, top_k=100)
