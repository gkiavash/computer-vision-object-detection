import csv
import itertools
import os
import imutils
import pickle
import cv2
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.object_detection import non_max_suppression

import config
import image_segmentation


def roi_ss(image):
    print("[INFO] running selective search...")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    proposals = []
    boxes = []
    for (x, y, w, h) in rects[:config.MAX_PROPOSALS_INFER]:
        roi = image[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        proposals.append(roi)
        boxes.append((x, y, x + w, y + h))

    proposals = np.array(proposals, dtype="float32")
    boxes = np.array(boxes, dtype="int32")
    print("[INFO] proposal shape: {}".format(proposals.shape))

    return proposals, boxes


def roi_seg(image, boxes, preview=False):
    boxes_ = []
    proposals = []
    for (startX, startY, endX, endY) in boxes:
        if endY-startY < 15 or endX-startX < 15:
            continue
        roi = image[startY:endY, startX:endX]
        roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)

        # if preview:
        #     cv2.imshow(str(len(proposals))+'___'+str(startX)+str(endY), roi)

        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        boxes_.append([startX, startY, endX, endY])
        proposals.append(roi)

    proposals = np.array(proposals, dtype="float32")
    boxes_ = np.array(boxes_, dtype="int32")
    print('boxes: ', boxes_.shape)
    print('proposals: ', proposals.shape)
    return proposals, boxes_


def roi_expand(boxes, range_min, range_max, step, w, h):
    def get_arr(startX__, range_min_, range_max_, max_boundary, step_):
        min_ = int(startX__ * range_min_)
        max_ = int(startX__ * range_max_)
        startX_ = [startX__]
        for i in range(min_, max_):
            if i % step_ == 0:
                if i > max_boundary:
                    break
                startX_.append(i)
        return startX_
    boxes_ = []
    for (startX, startY, endX, endY) in boxes:
        startX_ = get_arr(startX, range_min, range_max, w, step)
        startY_ = get_arr(startY, range_min, range_max, h, step)
        endX_ = get_arr(endX, range_min, range_max, w, step)
        endY_ = get_arr(startX, range_min, range_max, h, step)
        if len(startX_) == 0 or len(startY_) == 0 or len(endX_) == 0 or len(endY_) == 0:
            print(startX_)
            print(startY_)
            print(endX_)
            print(endY_)
            print()
        box_ = list(itertools.product(startX_, startY_, endX_, endY_))
        boxes_ += box_
    return boxes_


def detect(config, RESULT_MODEL_PATH, image, image_path, boxes, proposals, preview=False, save=False):
    print("[INFO] loading model and label binarizer from {}".format(RESULT_MODEL_PATH))
    model = load_model(RESULT_MODEL_PATH)
    lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())

    # image = imutils.resize(image, width=500)

    print("[INFO] classifying proposals...")
    proba = model.predict(proposals)
    print("[INFO] applying NMS...")
    labels = lb.classes_[np.argmax(proba, axis=1)]
    idxs = np.where(labels == config.OBJECT_TO_DETECT)[0]

    print(proba)

    boxes_ = []
    proba_ = []
    for i in range(len(proba)):
        if proba[i][0] > proba[i][1]:
            boxes_.append(boxes[i])
            proba_.append(proba[i][0])
    print('proba_: ', proba_)
    proba_ = np.array(proba_)
    boxes_ = np.array(boxes_)

    # boxes = boxes[idxs]
    # probb = proba[idxs][:, 0]
    # proba = proba[idxs][:, 1]
    # print('proba: ', proba)
    # print('probb: ', probb)

    # print("1: ", len(proba), "2: ", len(boxes_))
    # idxs = np.where(proba >= probb*10)
    # boxes = boxes[idxs]
    # proba = proba[idxs]

    # idxs = np.where(proba >= config.MIN_PROBA)
    # boxes = boxes[idxs]
    # proba = proba[idxs]

    clone = image.copy()
    for (box, prob) in zip(boxes_, proba_):
        print('box: ', box)
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        text = "{}: {:.2f}%".format(config.OBJECT_TO_DETECT, prob * 100)
        print(text)
        cv2.putText(clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    boxIdxs = non_max_suppression(boxes_, proba_)
    clone_nms = image.copy()
    for i in boxIdxs:
        # (startX, startY, endX, endY) = boxes[i]
        (startX, startY, endX, endY) = i
        cv2.rectangle(clone_nms, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        # text = "{}: {:.2f}%".format(config.OBJECT_TO_DETECT, proba[i] * 100)
        text = config.OBJECT_TO_DETECT
        cv2.putText(clone_nms, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    image_merged = cv2.vconcat([image, clone, clone_nms])
    if preview:
        cv2.imshow("image_merged", image_merged)
        cv2.waitKey()
    if save:
        full_path_img_image_merged = image_segmentation.image_path_new(image_path, 'RESULT')
        cv2.imwrite(full_path_img_image_merged, image_merged)
        print('SAVED: ', full_path_img_image_merged)

    return boxes_


def test_find_roi(image_path):
    file = image_path.split('/')[-1]
    file_name = file.split('.')[0]
    file_roi = file_name + '.csv'
    image_path_ = image_path.replace(file, file_roi)
    print(image_path_)

    boxes = []
    with open(image_path_) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == 'Index':
                continue
            startX = int(row[4])
            startY = int(row[5])
            endX = startX + int(row[6])
            endY = startY + int(row[7])
            boxes.append([startX, startY, endX, endY])
    print(boxes)
