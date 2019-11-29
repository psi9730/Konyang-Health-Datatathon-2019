import os
import argparse
import sys
import time
import random
import keras
import cv2
import numpy as np


from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import nsml
from nsml import DATASET_PATH, GPU_NUM, HAS_DATASET
from model import cnn_sample
from dataprocessing import resize_and_normalize, dataset_loader, RESIZED_HEIGHT, RESIZED_WIDTH

from keras_model import densenet
from keras_model import resnet
from keras import backend
from keras import layers
from keras import models
from keras import utils

from radam import RectifiedAdam

## setting values of preprocessing parameters
RESIZE = 10.
RESCALE = True

def wrapper():
        kwargs = dict()
        kwargs['backend'] = backend
        kwargs['layers'] = layers
        kwargs['models'] = models
        kwargs['utils'] = utils
        return kwargs

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        # model.save_weights(file_path,'model')
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')

    def infer(data, resized_height=RESIZED_HEIGHT, resized_width=RESIZED_WIDTH):  ## test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        X = []
        for i, d in enumerate(data):
            # test 데이터를 training 데이터와 같이 전처리 하기
            X.append(resize_and_normalize(d, resized_height, resized_width))
        X = np.array(X)

        pred = model.predict_classes(X)     # 모델 예측 결과: 0-3
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--epoch', type=int, default=10)                          # epoch 수 설정
    parser.add_argument('--batch_size', type=int, default=16)                      # batch size 설정
    parser.add_argument('--num_classes', type=int, default=4)                     # DO NOT CHANGE num_classes, class 수는 항상 4
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate (default: 0.001)')
    parser.add_argument('--train_valid_rate', type=float, default=0.85, help='ratio between train set and valid set')

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    parser.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    parser.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args = parser.parse_args()

    seed = 1234
    np.random.seed(seed)

    # training parameters
    nb_epoch = args.epoch
    batch_size = args.batch_size
    num_classes = args.num_classes

    """ Model """
    
    learning_rate = args.lr

    # model = cnn_sample(in_shape=(RESIZED_HEIGHT, RESIZED_WIDTH, 3), num_classes=num_classes)
    model = resnet.ResNet50(weights=None, in_shape = (RESIZED_HEIGHT, RESIZED_WIDTH, 3), num_classes=num_classes, **wrapper())

    adam = optimizers.Adam(lr=learning_rate, decay=1e-5)                    # optional optimization
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    optimizer = RectifiedAdam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

    bind_model(model)
    if args.pause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if args.mode == 'train':  ### training mode일 때
        print('Training Start...')

        img_path = DATASET_PATH + '/train/train_data/'
        if HAS_DATASET == False:
            img_path = DATASET_PATH + './sample_dataset/test'
        X_train, Y_train, X_val, Y_val = dataset_loader(img_path, args.train_valid_rate,
                                                                                resized_height=RESIZED_HEIGHT,
                                                                                resized_width=RESIZED_WIDTH)
        print(1)
        kwargs = dict(
            rotation_range=180,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True
        )
        print(2)
        train_datagen = ImageDataGenerator(**kwargs)
        print(3)
        train_generator = train_datagen.flow(x=X_train, y=Y_train, shuffle=True, batch_size=batch_size, seed=seed)
        print(4)
        # then flow and fit_generator....

        """ Callback """
        monitor = 'categorical_accuracy'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = len(X_train) // batch_size
        print('\n\nSTEP_SIZE_TRAIN = {}\n\n'.format(STEP_SIZE_TRAIN))

        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            print("### Model Fitting.. ###")
            print('epoch = {} / {}'.format(epoch+1, nb_epoch))
            print('check point = {}'.format(epoch))

            # for no augmentation case
            hist = model.fit_generator(train_generator,
                                       steps_per_epoch=len(X_train) / batch_size,
                                       validation_data=(X_val, Y_val),
                                       callbacks=[reduce_lr],
                                       )
            t2 = time.time()
            print(hist.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_acc = hist.history['categorical_accuracy'][0]
            train_loss = hist.history['loss'][0]
            val_acc = hist.history['val_categorical_accuracy'][0]
            val_loss = hist.history['val_loss'][0]

            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc, val_loss=val_loss, val_acc=val_acc)
            nsml.save(epoch)
        print('Total training time : %.1f' % (time.time() - t0))
        # print(model.predict_classes(X))



