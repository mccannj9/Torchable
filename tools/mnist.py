#! /usr/bin/env python3

"""
    Some utility functions for loading mnist data.

    The files along with instructions for reading the data can
    be found at the following address:

    http://yann.lecun.com/exdb/mnist/
    ---------------------------------
    train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
    train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
    t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
    t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

"""

import gzip
import numpy as np


def mnist_image_to_numpy(
    infile, max_images_to_load=None, dtype=np.float32
):
    """
        Convert the mnist images files (Yann LeCun's
        website) from gzipped files into numpy arrays

        infile (str) = full path to data file
        max_images_to_load (int) = number of images to load

        if max_images_to_load = None, whole file will be read

    """

    bytestream = gzip.open(infile)

    # All of this is specified on LeCun's website
    # throw away magic number of 2051
    int.from_bytes(bytestream.read(4), byteorder="big")

    # Allow user-specification of number of training images to load
    num_images = int.from_bytes(
        bytestream.read(4), byteorder="big"
    ) if not max_images_to_load else max_images_to_load

    num_rows = int.from_bytes(bytestream.read(4), byteorder="big")
    num_cols = int.from_bytes(bytestream.read(4), byteorder="big")

    # read in all of the image bytes and convert to np array
    num_bytes_to_read = num_images * num_rows * num_cols
    byte_data = bytearray(bytestream.read(num_bytes_to_read))
    images = np.array(byte_data).reshape(num_images, num_rows, num_cols)

    bytestream.close()

    return images.astype(dtype)


def mnist_label_to_numpy(
    infile, max_images_to_load=None,
    onehot_encode=True, dtype=np.float32
):
    """
        Convert the mnist labels files (Yann LeCun's
        website) from gzipped files into numpy arrays

        infile (str) = full path to data file
        max_images_to_load (int) = number of images to load
        onehot_encode (bool) = onehot encoding or integer label

        if max_images_to_load = None, whole file will be read

    """

    bytestream = gzip.open(infile)

    # throw away magic number of 2051
    int.from_bytes(bytestream.read(4), byteorder="big")  # 2051
    num_images = int.from_bytes(
        bytestream.read(4), byteorder="big"
    ) if not max_images_to_load else max_images_to_load

    # One byte per label
    labels = np.array(bytearray(bytestream.read(num_images)))
    bytestream.close()

    if onehot_encode:
        labels_onehot = np.zeros((labels.size, labels.max() + 1))
        labels_onehot[np.arange(labels.size), labels] = 1
        labels = labels_onehot

    return labels.astype(dtype)
