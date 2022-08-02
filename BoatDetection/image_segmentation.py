import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth

import config
from utils import image_path_new, colors_


def segment_watershed(image_path, preview=False):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    if preview:
        cv.imshow('ret', thresh)
        cv.waitKey(0)

    thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
    return thresh


def segment_meanshift(image, image_path_, kernel=1, preview=False, save=False):
    if image is None:
        image = cv.imread(image_path_)

    img_blured = image
    img_blured = cv.cvtColor(img_blured, cv.COLOR_BGR2HSV)
    flat_image = np.float32(img_blured.reshape((-1, 3)))

    bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)

    ms = MeanShift(bandwidth, max_iter=800, bin_seeding=True)
    ms.fit(flat_image)
    labeled = ms.labels_

    segments = np.unique(labeled)
    print('Number of blur: ', kernel)
    print('Number of segments: ', segments.shape[0])

    # # get the average color of each segment
    # total = np.zeros((segments.shape[0], 3), dtype=float)
    # count = np.zeros(total.shape, dtype=float)
    # for i, label in enumerate(labeled):
    #     total[label] = total[label] + flat_image[i]
    #     count[label] += 1
    # avg = total / count
    # avg = np.uint8(avg)
    # print('ave:', avg)
    # # cast the labeled image into the corresponding average color
    # res = avg[labeled]

    result = colors_(segments.shape[0])[labeled].reshape((image.shape))

    if preview:
        image_merged = cv.vconcat([image, img_blured, result])
        cv.imshow('meanshift'+str(kernel), image_merged)
        cv.waitKey()
    if save:
        image_merged = cv.vconcat([image, img_blured, result])
        cv.imwrite(image_path_new(image_path_, 'meanshift'+str(kernel)), image_merged)

    return result


def segment_kmean(image_path_, K=4, preview=False):
    img = cv.imread(image_path_)
    Z = np.float32(img.reshape((-1, 3)))
    print('Z: ', Z.shape)

    img_coord = []
    for i in range(len(img)):
        for j in range(len(img[i])):
            img_coord.append([i, j])
    img_coord = np.array(img_coord, dtype="float32")
    print(img_coord.shape)
    Z_ = np.concatenate((Z, img_coord), axis=1)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv.kmeans(Z_, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    print('_: ', _)
    print('labels: ', labels)
    print('centers: ', centers)

    segmented_data = colors_(K)[labels.flatten()]
    segmented_image_ = np\
        .array(segmented_data, dtype="float32")\
        .reshape((img.shape))

    if preview:
        cv.imshow('base', img)
        cv.imshow('ret', segmented_image_)
        cv.waitKey()
    return segmented_image_


def find_countours(image, image_path, kernel=5, preview=False, save=False):
    layers = cv.split(image)
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def get_contours(image_1_layer):
        contours1, hierarchy1 = cv.findContours(
            image=image_1_layer,
            mode=cv.RETR_TREE,
            method=cv.CHAIN_APPROX_SIMPLE
        )
        # print('contours1: ', contours1)
        # print('hierarchy1: ', hierarchy1)

        image_contour = image.copy()
        cv.drawContours(
            image=image_contour,
            contours=contours1,
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv.LINE_AA
        )
        return image_contour, contours1, hierarchy1

    def get_boxes(image_, contours):
        boxes = []
        for each_rect in contours:
            startX=10000; startY=10000; endX=0; endY=0
            for each_coord in each_rect:
                startX = min(each_coord[0][0], startX)
                startY = min(each_coord[0][1], startY)
                endX = max(each_coord[0][0], endX)
                endY = max(each_coord[0][1], endY)
            cv.rectangle(image_, (startX, startY), (endX, endY), (255, 255, 0), 2)
            boxes.append([startX, startY, endX, endY])
        return boxes

    boxes = []
    images_ = []
    for i in layers:
        im, co, hi = get_contours(i)
        boxes += get_boxes(im, co)
        images_.append(im)

    print('Image Segmented with kernel: {}'.format(kernel))
    if preview:
        image_merged_layers = cv.vconcat(layers)
        image_merged_images = cv.vconcat(images_)
        cv.imshow('Splited'+str(kernel), image_merged_layers)
        cv.imshow('Contour'+str(kernel), image_merged_images)
        cv.waitKey()

    if save:
        image_merged_layers = cv.vconcat(layers)
        image_merged_images = cv.vconcat(images_)
        cv.imwrite(image_path_new(image_path, 'Splited'+str(kernel)), image_merged_layers)
        cv.imwrite(image_path_new(image_path, 'Contour'+str(kernel)), image_merged_images)

    return boxes
