import os
import argparse
import sys
import time
import pickle
import random
import keras
import cv2
import numpy as np


# RESIZED_HEIGHT = 650
# RESIZED_WIDTH = 512

RESIZED_HEIGHT = 390
RESIZED_WIDTH = 307


def resize_and_normalize(im, resized_height=RESIZED_HEIGHT, resized_width=RESIZED_WIDTH):
    # h, w, c = 3900, 3072, 3
    # nh, nw = int(h//resize_factor), int(w//resize_factor)
    im = cv2.resize(im, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    im = cv2.addWeighted(im, 4, cv2.GaussianBlur(im, (0, 0), resized_height/60), -4, 128)
    im = im / 255.
    return im  # 0 - 255 사이 값


def Label2Class(label):     # one hot encoding (0-3 --> [., ., ., .])
    # label smoothing
    resvec = [0.03333, 0.03333, 0.03333, 0.03333]
    if label == 'AMD':
        cls = 1
        resvec[cls] = 0.9
    elif label == 'RVO':
        cls = 2
        resvec[cls] = 0.9
    elif label == 'DMR':
        cls = 3
        resvec[cls] = 0.9
    else:  # Normal
        cls = 0
        resvec[cls] = 0.9
    return resvec


def dataset_loader(train_path, train_valid_rate, resized_height, resized_width):
    t1 = time.time()
    print('Loading training data...\n')

    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []

    categories = ['NOR', 'AMD', 'RVO', 'DMR']
    for category in categories:
        category_path = os.path.join(train_path, category + '/')
        filenames = os.listdir(category_path)
        train_files_num = int(len(filenames) * train_valid_rate)
        if train_files_num % 2 != 0:  # train_files_num 항상 짝수가 되게 -> 같은사람의 L, R 나뉘지 않게
            train_files_num += 1
        for index, filename in enumerate(filenames):
            # rgb 순서 제대로 읽혔나 확인 필요
            im = cv2.imread(os.path.join(category_path, filename))
            im = resize_and_normalize(im, resized_height, resized_width)
            if index < train_files_num:
                train_images.append(im)
                train_labels.append(Label2Class(category))
            else:
                valid_images.append(im)
                valid_labels.append(Label2Class(category))
            del im
            print(index + 1, '/', filename, ' image(s)')

    #
    #     with open('./' + category + 'train.pkl', 'wb') as fp:
    #         pickle.dump(train_images, fp)
    #     with open('./' + category + 'train_label.pkl', 'wb') as fp:
    #         pickle.dump(train_labels, fp)
    #     with open('./' + category + 'valid.pkl', 'wb') as fp:
    #         pickle.dump(valid_images, fp)
    #     with open('./' + category + 'valid_label.pkl', 'wb') as fp:
    #         pickle.dump(valid_labels, fp)
    #
    #     train_images = []
    #     train_labels = []
    #     valid_images = []
    #     valid_labels = []
    #
    # for category in categories:
    #     with open('./' + category + 'train.pkl', 'rb') as fp:
    #         temp_array = pickle.load(fp)
    #     train_images += temp_array
    #     with open('./' + category + 'train_label.pkl', 'rb') as fp:
    #         temp_array = pickle.load(fp)
    #     train_labels += temp_array
    #     with open('./' + category + 'valid.pkl', 'rb') as fp:
    #         temp_array = pickle.load(fp)
    #     valid_images += temp_array
    #     with open('./' + category + 'valid_label.pkl', 'rb') as fp:
    #         temp_array = pickle.load(fp)
    #     valid_labels += temp_array

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    valid_images = np.array(valid_images)
    valid_labels = np.array(valid_labels)

    t2 = time.time()

    print('Dataset prepared for', t2 - t1, 'sec')
    print('Train Images:', train_images.shape, 'np.array.shape(files, views, width, height)')
    print('Train Labels:', train_labels.shape, ' among 0-3 classes')
    print('Valid Images:', valid_images.shape, 'np.array.shape(files, views, width, height)')
    print('Valid Labels:', valid_labels.shape, ' among 0-3 classes')

    return train_images, train_labels, valid_images, valid_labels



