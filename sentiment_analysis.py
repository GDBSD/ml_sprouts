# -*- coding: utf-8 -*-
import os
import logging

import helpers.data_helpers as fetchers

ROOT = os.path.abspath(os.path.join(".", os.pardir))

# Set the Keras data cache path. This is specific to the level in the project for a module.
KERAS_CACHE_DIR = '{}/ml_sprouts/big_data/datasets/'.format(ROOT)


logging.basicConfig(level=logging.DEBUG)


class ImdbSentimentAnalysis:

    def analyze_data(self, data_source, data_folder, file_name):
        imdm_data = self.get_data(data_source=data_source, data_folder=data_folder,
                                  file_name=file_name)

    @staticmethod
    def get_data(data_source, data_folder, file_name):
        data = fetchers.load_binary_text_classifier_dataset(data_source=data_source,
                                                            data_folder=data_folder,
                                                            file_name=file_name)
        train_texts = data[0]
        test_texts = data[1]
        logging.info('Returning {} training records and {} test records.'.format(
            len(train_texts), len(test_texts)))

        return data


if __name__ == "__main__":
    imdb_data_source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    imdb_data_folder = '{}aclimdb'.format(KERAS_CACHE_DIR)
    filename = 'aclImdb_v1.tar.gz'
    analyzer = ImdbSentimentAnalysis()
    analyzer.analyze_data(data_source=imdb_data_source, data_folder=imdb_data_folder, file_name=filename)
