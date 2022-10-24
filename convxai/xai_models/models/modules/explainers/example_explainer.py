import json
import h5py
import torch.nn as nn
import numpy as np


class ExampleExplainer(object):
    
    def __init__(self, conference,  model_configs_dir = "/data/hua/workspace/projects/convxai/src/convxai/xai_models/configs/xai_model_configs.json"):
        super().__init__()

        with open(model_configs_dir, 'r') as f:
            self.model_configs = json.load(f)


        self.diversity_model_embeddings_h5dir = self.model_configs["xai_emample_embeddings_dir"][conference]
        self.diversity_model_texts_h5dir = self.model_configs["xai_emample_texts_dir"][conference]


        with h5py.File(self.diversity_model_embeddings_h5dir, 'r') as infile:
            self.diversity_x_train_embeddings_tmp = np.empty(infile["x_train_embeddings"].shape, dtype=infile["x_train_embeddings"].dtype)
            infile["x_train_embeddings"].read_direct(self.diversity_x_train_embeddings_tmp)


        with h5py.File(self.diversity_model_texts_h5dir, 'r') as infile:
            self.diversity_x_train_text_tmp = np.empty(infile["x_train_text"].shape, dtype=infile["x_train_text"].dtype)
            infile["x_train_text"].read_direct(self.diversity_x_train_text_tmp)
            self.diversity_x_train_title_tmp = np.empty(infile["x_train_title"].shape, dtype=infile["x_train_title"].dtype)
            infile["x_train_title"].read_direct(self.diversity_x_train_title_tmp)
            self.diversity_x_train_link_tmp = np.empty(infile["x_train_link"].shape, dtype=infile["x_train_link"].dtype)
            infile["x_train_link"].read_direct(self.diversity_x_train_link_tmp)


            self.diversity_aspect_list_tmp = np.empty(infile["aspect_list"].shape, dtype=infile["aspect_list"].dtype)
            infile["aspect_list"].read_direct(self.diversity_aspect_list_tmp)
            self.diversity_aspect_confidence_list_tmp = np.empty(infile["aspect_confidence_list"].shape, dtype=infile["aspect_confidence_list"].dtype)
            infile["aspect_confidence_list"].read_direct(self.diversity_aspect_confidence_list_tmp)
            self.diversity_perplexity_tmp = np.empty(infile["perplexity"].shape, dtype=infile["perplexity"].dtype)
            infile["perplexity"].read_direct(self.diversity_perplexity_tmp)
            self.diversity_token_counts_tmp = np.empty(infile["token_counts"].shape, dtype=infile["token_counts"].dtype)
            infile["token_counts"].read_direct(self.diversity_token_counts_tmp)


