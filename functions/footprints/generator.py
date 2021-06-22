import numpy as np
import random
import keras
import json
from keras.preprocessing.image import ImageDataGenerator


class SatelliteImageDataGenerator(keras.utils.Sequence):
    """ Data generator used to randomly sample large satellite image. Stores band-wise mean and std for image
    centering. """

    def __init__(self, img, labels, img_size, batch_size, n_samples):
        self.img = img
        self.img_size = img_size
        self.labels = labels
        self.batch_size = batch_size
        self.n_samples = n_samples

        self.width = img.shape[1]
        self.height = img.shape[0]
        self.indices = set()

        self.mean = np.zeros(img.shape[2])
        self.std = np.zeros(img.shape[2])

    def __len__(self):
        return (np.ceil(self.n_samples / float(self.batch_size))).astype(np.int)

    def __getitem__(self, index):
        X_batch = []
        y_batch = []
        for i in range(self.batch_size):
            i_w = np.random.randint(self.width - self.img_size)
            i_h = np.random.randint(self.height - self.img_size)
            while ((i_w,
                    i_h) in self.indices):  # repeat until fresh indices are generated
                i_w = np.random.randint(self.width - self.img_size)
                i_h = np.random.randint(self.height - self.img_size)
            self.indices.add((i_w, i_h))
            sample = self.img[i_h:(i_h + self.img_size),
                     i_w:(i_w + self.img_size)]  # take img_size x img_size crop of full image with
            label = self.labels[i_h:(i_h + self.img_size), i_w:(i_w + self.img_size)]

            # feature-wise normalization/centering
            sample = self.flow(sample)
            X_batch.append(sample)
            y_batch.append(label)
        return np.array(X_batch), np.array(y_batch)

    def fit(self, img_fit):  # fit datagen to featurewise statistics of img_fit
        mean = []
        std = []
        for i in range(self.img.shape[2]):
            mean.append(np.mean(img_fit[:, :, i]))
            std.append(np.std(img_fit[:, :, i]))
        self.mean = np.array(mean)
        self.std = np.array(std)

    def flow(self, X):  # process one sample X
        sample = np.copy(X)
        for i in range(X.shape[2]):
            sample[:, :, i] = (X[:, :, i] - self.mean[i]) / self.std[i]
        return sample

    def to_json(self, path):
        d = {'mean': list(self.mean.astype(float)), 'std': list(self.std.astype(float))}
        with open(path, 'w') as file:
            json.dump(d, file)
        print('serialized generator constants to JSON')


#
class MultiSatelliteImageDataGenerator(keras.utils.Sequence):
    """ Multi-image version of SatelliteImageDataGenerator. Randomly chooses between both samplers each item of
    batch. """

    def __init__(self, generators, batch_size, n_samples, weights=None):
        self.generators = generators
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.weights = weights

    def __len__(self):
        return (np.ceil(self.n_samples / float(self.batch_size))).astype(np.int)

    def __getitem__(self, index):
        X_batch = []
        y_batch = []
        for i in range(self.batch_size):
            gen = random.choices(self.generators, weights=self.weights)[0]  # choose a random generator
            item = gen.__getitem__(index)  # for now assumes batch size of generators is 1
            X_batch.append(item[0][0])
            y_batch.append(item[1][0])
        return np.array(X_batch), np.array(y_batch)
