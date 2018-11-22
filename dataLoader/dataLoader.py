# coding:utf-8

import random
import pickle
import os
import numpy as np
from Dataset import Document, Dataset
from Vocab import Vocab

class BatchDataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=True):
        assert isinstance(dataset, Dataset)
        assert len(dataset) >= batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.dataset(self.batch_size, self.shuffle))


class PickleReader():
    """
    this class intends to read pickle files converted by RawReader
    """

    def __init__(self, pickle_data_dir="../data/CNN_DM_pickle_data/"):
        """
        :param pickle_data_dir: the base_dir where the pickle data are stored in
        this dir should contain train.p, val.p, test.p, and vocab.p
        this dir should also contain the chunked_data folder
        """
        self.base_dir = pickle_data_dir

    def data_reader(self, dataset_path):
        """
        :param dataset_path: path for data.p
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        return data

    def full_data_reader(self, dataset_type="train"):
        """
        this method read the full dataset
        :param dataset_type: "train", "val", or "test"
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        return self.data_reader(self.base_dir + dataset_type + ".p")

    def chunked_data_reader(self, dataset_type="train", data_quota=-1):
        """
        this method reads the chunked data in the chunked_data folder
        :return: a iterator of chunks of datasets
        """
        data_counter = 0
        # chunked_dir = self.base_dir + "chunked/"
        chunked_dir = os.path.join(self.base_dir, 'chunked')
        os_list = os.listdir(chunked_dir)
        if data_quota == -1: #none-quota randomize data
            random.seed()
            random.shuffle(os_list)

        for filename in os_list:
            if filename.startswith(dataset_type):
                # print("filename:", filename)
                chunk_data = self.data_reader(os.path.join(chunked_dir, filename))
                if data_quota != -1:  # cut off applied
                    quota_left = data_quota - data_counter
                    # print("quota_left", quota_left)
                    if quota_left <= 0:  # no more quota
                        break
                    elif quota_left > 0 and quota_left < len(chunk_data):  # return partial data
                        yield Dataset(chunk_data[:quota_left])
                        break
                    else:
                        data_counter += len(chunk_data)
                        yield chunk_data
                else:
                    yield chunk_data
            else:
                continue

