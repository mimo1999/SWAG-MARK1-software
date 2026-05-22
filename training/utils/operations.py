import numpy as np


def extract_crops(img, crop_height, crop_width, step_vertical=None, step_horizontal=None):
    img_height, img_width = img.shape[:2]
    crop_height = min(crop_height, img_height)
    crop_width = min(crop_width, img_width)

    crops = []
    crops_boxes = []

    if not step_horizontal:
        step_horizontal = crop_width
    if not step_vertical:
        step_vertical = crop_height

    height_offset = 0
    last_row = False
    while not last_row:
        if img_height - height_offset < crop_height:
            height_offset = img_height - crop_height
            last_row = True
        last_column = False
        width_offset = 0
        while not last_column:
            if img_width - width_offset < crop_width:
                width_offset = img_width - crop_width
                last_column = True
            ymin, ymax = height_offset, height_offset + crop_height
            xmin, xmax = width_offset, width_offset + crop_width
            a_crop = img[ymin:ymax, xmin:xmax]
            crops.append(a_crop)
            crops_boxes.append((ymin, xmin, ymax, xmax))
            width_offset += step_horizontal
        height_offset += step_vertical
    return np.stack(crops, axis=0), crops_boxes


def get_absolute_boxes(box_absolute, boxes_relative):
    absolute_boxes = []
    absolute_ymin, absolute_xmin, _, _ = box_absolute
    for relative_box in boxes_relative:
        absolute_boxes.append(relative_box + [absolute_ymin, absolute_xmin, absolute_ymin, absolute_xmin])
    return absolute_boxes


def non_max_suppression_fast(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")
