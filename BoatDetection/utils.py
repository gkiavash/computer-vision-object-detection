import os
import cv2
import numpy as np
import config


def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def evaluate_iou(boxes_ground_truth, boxes_predict):
    evals = []
    for box_ in boxes_ground_truth:
        eval = [compute_iou(box_, box_p) for box_p in boxes_predict]
        print(eval)
        print(max(eval))
        evals.append(max(eval))
    return evals


def evaluate_iou_ave(eval_path):
    json_data = {}
    for index, filename in enumerate(os.listdir(eval_path)):
        full_path = os.path.join(eval_path, filename)

        text_file = open(full_path, "r")
        op = text_file.read().strip('][').split(', ')
        print(op)
        try:
            op = [float(i) for i in op]
        except:
            print(filename)
            continue
        json_data.update({filename: round(sum(op) / len(op), 2)})
    return json_data
    s = "C:/Users/ASUS/Desktop/Project/FP_ml/BoatDetection/results/final_evaluation"


def image_path_new(image_path, extension):
    file_name_ex = image_path.split('/')[-1]
    file_name = file_name_ex.split('.')[0]
    ex = '.' + file_name_ex.split('.')[1]
    print(config.RESULT_IMAGE_PATH)
    full_path_img_image_merged = config.RESULT_IMAGE_PATH + '/' + file_name + extension + ex
    print('NEW PATH: ', full_path_img_image_merged)
    return full_path_img_image_merged


def colors_(number_needed):
    all = [[0, 0, 0]]
    for i in range(1, int(number_needed/7)+2):
        x = i*30
        x = 255 - x
        all += [
            [x, 0, 0],
            [0, x, 0],
            [0, 0, x],

            [x, x, 0],
            [0, x, x],
            [x, 0, x],

            [x, x, x],
        ]
    return np.uint8(all)


def equalizeHist_(image, image_path, preview=False, save=False):
    if image is None:
        image = cv2.imread(image_path)

    channels = cv2.split(image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channels_eq = []
    for i in channels:
        channels_eq.append(cv2.equalizeHist(i))
    image_eq = cv2.merge(channels_eq)

    print('Image Equalized!')
    if preview:
        image_merged = cv2.vconcat([image, image_eq])
        cv2.imshow('Eq', image_merged)
        cv2.waitKey()
    if save:
        image_merged = cv2.vconcat([image, image_eq])
        cv2.imwrite(image_path_new(image_path, 'Eq'), image_merged)
    return image_eq


def dilation_(image, image_path, kernel, preview=False, save=False):
    if image is None:
        image = cv2.imread(image_path)

    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * kernel + 1, 2 * kernel + 1),
        (kernel, kernel)
    )
    image_dilatation = cv2.dilate(image, element)
    if preview:
        image_merged = cv2.vconcat([image, image_dilatation])
        cv2.imshow('dilation_dilation_', image_merged)
        cv2.waitKey()
    if save:
        image_merged = cv2.vconcat([image, image_dilatation])
        cv2.imwrite(image_path_new(image_path, 'Dil'), image_merged)
    return image_dilatation


def erosion_(image, image_path, kernel, post_name, preview=False, save=False):
    if image is None:
        image = cv2.imread(image_path)

    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * kernel + 1, 2 * kernel + 1),
        (kernel, kernel)
    )
    image_dilatation = cv2.erode(image, element)
    if preview:
        image_merged = cv2.vconcat([image, image_dilatation])
        cv2.imshow('dilation_dilation_', image_merged)
        cv2.waitKey()
    if save:
        image_merged = cv2.vconcat([image, image_dilatation])
        cv2.imwrite(image_path_new(image_path, 'Ero{}'.format(post_name)), image_merged)
    return image_dilatation


def remove_additional_dataset(k):
    import glob
    import os
    f = glob.glob("C:/Users/ASUS/Desktop/Project/FP_ml/BoatDetection/training_data/no_boat/*")
    for i in range(len(f)):
        if i % k == 1:
            os.remove(f[i])

    f = glob.glob("C:/Users/ASUS/Desktop/Project/FP_ml/BoatDetection/training_data/no_boat/*")
    for i in range(len(f)):
        if i % k == 1:
            os.remove(f[i])
