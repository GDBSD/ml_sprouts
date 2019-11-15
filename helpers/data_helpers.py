# -*- coding: utf-8 -*-

import logging
import os
import os.path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)

ROOT = os.path.abspath(os.path.join(".", os.pardir))
KERAS_CACHE_DIR = '{}/big_data/datasets'.format(ROOT)

# Make functions deterministic
RANDOM_SEED = 42


def load_binary_text_classifier_dataset(data_source, data_folder, file_name):
    """Loads a dataset that contains text and labels. If the dataset
    isn't already in the cache tf.keras.utils.get_file() downloads it.

    # Arguments
        data_source: string, path to the data directory.
        data_folder: string, name of the folder Keras will create
        file_name: string, name of the file

    # Returns
        A tuple of training and validation data.
        Number of categories: 2 (0 - negative, 1 - positive)
    """

    # This will download the dataset if it's not already in the cache.
    logging.info('Loading data from {} .....'.format(data_folder))
    tf.keras.utils.get_file(fname=file_name,
                            origin=data_source,
                            cache_dir=data_folder,
                            extract=True)

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = '{}/train/{}'.format(data_folder, category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = '{}/test/{}'.format(data_folder, category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    random.seed(RANDOM_SEED)
    random.shuffle(train_texts)
    random.shuffle(train_labels)

    logging.info('Returning {} training records and {} test records.'.format(
        len(train_texts), len(test_texts)))

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))


def create_df(data, col_names):
    df = pd.DataFrame(list(zip(data[0], data[1])),
                      columns=col_names)
    return df


if __name__ == "__main__":
    imdb_data_source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    imdb_data_folder = '{}/aclimdb'.format(KERAS_CACHE_DIR)
    filename = 'aclImdb_v1.tar.gz'

    train_test_data = load_binary_text_classifier_dataset(
        data_source=imdb_data_source,
        data_folder=imdb_data_folder,
        file_name=filename)

