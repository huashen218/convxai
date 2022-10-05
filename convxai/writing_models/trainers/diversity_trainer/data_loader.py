import os
import h5py
import json
import torch
import numpy as np
from torch.utils import data

from transformers import (
    BertTokenizerFast
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class PredDataset(data.Dataset):
    """For AspectPredict Writing Model
    """
    def __init__(self, x, bucket_num=None):
        self.x = x.astype(np.int64)

    def __len__(self):
        return self.x.shape[0] 

    def __getitem__(self, index):
        return self.x[index]



class CovidDataset(data.Dataset):
    """For AspectPredict Writing Model
    """
    def __init__(self, x, y, bucket_num=None):
        self.x = x.astype(np.int64)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.y.shape[0] 

    def __getitem__(self, index):
        return self.x[index], self.y[index]




class Feature:
    def __init__(self, pad_length=100, tokenizer=None):
        self.pad_length = pad_length
        if tokenizer is None:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        self.pad_id, self.cls_id, self.sep_id = self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]
            )

    def extract(self, sents):
        results = [self.tokenizer.encode(s, add_special_tokens=False) for s in sents]
        # results = [self.tokenizer.encode(sents, add_special_tokens=False)]

        # turn to matrix with padding
        matrix = np.ones([len(results), self.pad_length], dtype=np.int32) * self.pad_id
        for i, res in enumerate(results):
            length = min(len(res), self.pad_length)
            matrix[i, :length] = res[:length]

        cls_matrix = np.ones([len(results), 1]) * self.cls_id
        sep_matrix = np.ones([len(results), 1]) * self.sep_id
        matrix = np.hstack([cls_matrix, matrix, sep_matrix])
        
        return matrix




def load_raw_data(data_dir, phrase="train", verbose=False):

    filenames = [f for f in os.listdir(os.path.join(data_dir, phrase)) if ".swp" not in f and ".json" in f]
    if verbose:
        print("Loading {} data".format(phrase))
        print("# file", len(filenames))

    x = []
    y = []
    for filename in filenames:
        with open(os.path.join(data_dir, phrase, filename), 'r', encoding='utf-8') as infile:
            data = json.load(infile)
            x.extend([
                segment["segment_text"]
                for paragraph in data["abstract"]
                for sent in paragraph["sentences"]
                for segment in sent
            ])
            y.extend([
                segment["crowd_label"]
                for paragraph in data["abstract"]
                for sent in paragraph["sentences"]
                for segment in sent
            ])
    
    if verbose:
        print("# x", len(x))
        print("# y", len(y))

        # stat for x
        x_stat = np.array([len(xx.split()) for xx in x])
        print("avg seq length = {:.3f} (SD={:.3f})".format(x_stat.mean(), x_stat.std()))
        print("min = {}, max = {}".format(x_stat.min(), x_stat.max()))

        print()

    return x, y




def h5_load(filename, data_list, dtype=None, verbose=False):
    with h5py.File(filename, 'r') as infile:
        data = []
        for data_name in data_list:
            if dtype is not None:
                temp = np.empty(infile[data_name].shape, dtype=dtype)
            else:
                temp = np.empty(infile[data_name].shape, dtype=infile[data_name].dtype)
            infile[data_name].read_direct(temp)
            data.append(temp)
          
        if verbose:
            print("\n".join(
                "{} = {} [{}]".format(data_name, str(real_data.shape), str(real_data.dtype))
                for data_name, real_data in zip(data_list, data)
            ))
            print()
        return data





def data_loader(configs, cache_dir, tokenizer):

    data_cached_path = os.path.join(cache_dir, configs["model_version"]+".h5")
    covid_data_dir = os.path.join(configs["save_dirs"]["root_dir"], configs["save_dirs"]["covid_data_dir"])
    label_mapping = configs["data_params"]["label_mapping"]

    if os.path.isfile(data_cached_path):
        x_train, y_train, x_test, y_test, x_dev, y_dev = h5_load(data_cached_path, [
            "x_train", "y_train", "x_test", "y_test", "x_dev", "y_dev"
        ], dtype=np.int32, verbose=True)

    else:
        # load data
        x_train, y_train = load_raw_data(covid_data_dir, phrase="train", verbose=True)
        x_test, y_test = load_raw_data(covid_data_dir, phrase="test", verbose=True)
        x_dev, y_dev = load_raw_data(covid_data_dir, phrase="dev", verbose=True)
        
        feature = Feature(tokenizer=tokenizer)
        x_train = feature.extract(x_train[:])
        x_test = feature.extract(x_test[:])
        x_dev = feature.extract(x_dev[:])

        # turn label into vector
        y_train = np.array([label_mapping[y] for y in y_train])
        y_test = np.array([label_mapping[y] for y in y_test])
        y_dev = np.array([label_mapping[y] for y in y_dev])

        # cache data
        with h5py.File(data_cached_path, 'w') as outfile:
            outfile.create_dataset("x_train", data=x_train) 
            outfile.create_dataset("y_train", data=y_train)
            outfile.create_dataset("x_test", data=x_test) 
            outfile.create_dataset("y_test", data=y_test)
            outfile.create_dataset("x_dev", data=x_dev) 
            outfile.create_dataset("y_dev", data=y_dev)

    print("Train", x_train.shape, y_train.shape)
    print("Test", x_test.shape, y_test.shape)
    print("Valid", x_dev.shape, y_dev.shape)

    train_dataset = CovidDataset(x_train, y_train)
    test_dataset = CovidDataset(x_test, y_test)
    dev_dataset = CovidDataset(x_dev, y_dev)
    train_dataloader = data.DataLoader(train_dataset, batch_size=configs["model_params"]["scibert_param"]["batch_size"], shuffle=True, num_workers=4)
    test_dataloader = data.DataLoader(test_dataset, batch_size=configs["model_params"]["scibert_param"]["batch_size"], shuffle=False, num_workers=4)
    dev_dataloader = data.DataLoader(dev_dataset, batch_size=configs["model_params"]["scibert_param"]["batch_size"], shuffle=False, num_workers=4)

    return train_dataloader, dev_dataloader, test_dataloader

