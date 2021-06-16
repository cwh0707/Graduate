from __future__ import print_function
from os import listdir
from os.path import isfile, join
import numpy as np
from skimage.io import imread
from skimage.transform import rotate
from keras.utils import to_categorical
from random import shuffle, randint
from data import image_shape, compute_statistics_file
import sys


class UNetGeneratorClass(object):
    """
    It creates a generator to train/validate the model
    """
    def __init__(self, data_path, n_class=5, batch_size=32, channels=3, apply_augmentation=True, thres_score=None,
                 train = True):
        """
        Initialization of the generator 生成器初始化
        :param n_class: number of classes to predict 预测类别
        :param batch_size: size of the batch for training the model 每个块的大小
        :param apply_augmentation: if True, data augmentation is applied
        :param thres_score: value between 0 and 1. Minimum ratio of lesion that images of the generator will have
        :param train: boolean value. If true, the generator takes the train images; if false, the generator takes
        the validation images
        """

        if train:
            path = data_path + 'train/'
        else:
            path = data_path + 'val/'

        self.labels_path = join(path, 'labels/')
        self.image_path = join(path, 'images/')
        self.path = path

        self.files = [f for f in listdir(self.image_path) if isfile(join(self.image_path, f)) and f.endswith('.png') ]

        if thres_score is not None:
            files_aux = []
            for file_name in self.files:
                a, b, c = file_name.split("_")
                label_img = imread(self.labels_path + a + '_' + b + '_label_' + c)
                score = compute_statistics_file(label_img)
                if score > thres_score:
                    files_aux.append(file_name)
            self.files = files_aux

        self.img_shape = image_shape(self.image_path + self.files[0])
        self.channels = channels

        self.batch_size = batch_size
        self.n_class = n_class

        self.apply_augmentation = apply_augmentation

    def data_augmentation(self, data, delta):
        """
        It applies a random trasnformation to the data
        :param data: data to apply the transformation
        :param delta: random value between 0 and 3
        :return: the transformed data
        用于数据增强
        """
        # 对图像进行水平翻转 本质就是左右交换
        h_i_mirror = np.fliplr(data)
        # 对图像进行垂直翻转 就是上下交换
        v_i_mirror = np.flipud(data)

        if delta == 0:
            return data
        elif delta == 1:
            return h_i_mirror
        elif delta == 2:
            return v_i_mirror
        elif delta == 3:
            return rotate(data, angle=180)

    def generate(self):
        """
        It yields images and labels for every batch to train the model
        """
        while True:

            shuffle(self.files)

            for n_batch in range(len(self.files) // self.batch_size):
                batch_files = self.files[n_batch*self.batch_size:(n_batch+1)*self.batch_size]

                images_batch = np.zeros((self.batch_size, self.img_shape[0], self.img_shape[1], self.channels), dtype=np.uint8)
                labels_batch = np.zeros((self.batch_size, self.img_shape[0], self.img_shape[1], self.n_class), dtype=np.uint8)

                n = 0
                for file_name in batch_files:

                    img = imread(self.image_path + file_name)
                    image_array = np.asarray(img).astype(np.uint8)

                    a, b, c = file_name.split("_")

                    label_name = a + '_' + b + '_label_' + c
                    label_img = imread(self.labels_path + label_name)
                    label_array = np.asarray(label_img[:, :, 0]).astype(np.uint8)
                    assert (np.amin(label_array) >= 0 and np.amax(label_array) <= 5)

                    if self.apply_augmentation:
                        delta = randint(0, 3)
                        img_augm = self.data_augmentation(image_array, delta)
                        images_batch[n, :, :, :] = img_augm
                        images_batch = np.asarray(images_batch).astype(np.float32)
                        labels_augm = self.data_augmentation(label_array, delta)
                        labels_batch[n, :, :, :] = to_categorical(labels_augm, self.n_class).reshape(labels_augm.shape + (self.n_class,))
                        labels_batch = np.asarray(labels_batch).astype(np.float32)
                    else:
                        images_batch[n, :, :, :] = image_array
                        images_batch = np.asarray(images_batch).astype(np.float32)
                        # test = to_categorical(label_array, self.n_class)
                        # print(label_array.shape)
                        # print(label_array.shape + (self.n_class,))
                        labels_batch[n, :, :, :] = to_categorical(label_array, self.n_class).reshape(label_array.shape + (self.n_class,))
                        labels_batch = np.asarray(labels_batch).astype(np.float32)
                    n += 1
                # 生成一对图像和对应的mask
                yield (images_batch, labels_batch)