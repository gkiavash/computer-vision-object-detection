###############################################################################
# building training dataset from raw images and annotations
# comment this import after "training_data" directory is filled
# import build_dataset

###############################################################################
# training neural network from "boat" and "no_boat" data in "training_data"
# comment this import after "results/model" directory is filled
# import fine_tune_rcnn

###############################################################################
# detection: all code are pu in a function for easier runtime management
import os
import cv2
import csv

import config
from detect_object_rcnn_min import *
from utils import *
from image_segmentation import *


def detect_object_rcnn(model_path):
    annotation_paths = os.listdir(config.TEST_ANNOTS)

    # for F_ in range(len(image_paths)):
    #     F = image_paths[F_]

    for index, filename in enumerate(os.listdir(config.TEST_IMAGES)):
        print(os.path.join(config.TEST_IMAGES, filename))
        F = os.path.join(config.TEST_IMAGES, filename).replace('\\', '/')

        boxes = []
        # image = equalizeHist_(image=None, image_path=F, preview=False)
        image = cv2.imread(F)

        for kernel in [5, 45, 65]:
            image_blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
            image_blurred = segment_meanshift(image=image_blurred, image_path_=F, kernel=kernel, preview=False, save=True)
            # image_blurred = erosion_(image=image_blurred, image_path=F, kernel=5, post_name=kernel,preview=False, save=True)
            boxes += find_countours(image=image_blurred, image_path=F, kernel=kernel, preview=False, save=True)

        proposals, boxes = roi_seg(image=image, boxes=boxes, preview=False)
        boxes_predicted = detect(
            config=config,
            RESULT_MODEL_PATH=model_path,
            image=image,
            image_path=F,
            boxes=boxes,
            proposals=proposals,
            preview=False,
            save=True
        )
        boxes_ground_truth = []
        with open(os.path.join(config.TEST_ANNOTS, annotation_paths[index]), 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                try:
                    int(row[0])
                except:
                    continue

                boxes_ground_truth.append([
                    int(row[4]),
                    int(row[5]),
                    int(row[6]) + int(row[4]),
                    int(row[7]) + int(row[5])
                ])
        print(boxes_ground_truth)
        print('EVAL: ', evaluate_iou(boxes_ground_truth, boxes_predicted))

        f = open(os.path.join(config.TEST_ANNOTS, filename+".txt"), "a")
        f.write(str(evaluate_iou(boxes_ground_truth, boxes_predicted)))
        f.close()


detect_object_rcnn(config.RESULT_MODEL_PATH_MobileNet)


###############################################################################
# To freeze the model for C++
# import h5_to_pb
