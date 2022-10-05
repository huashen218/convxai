import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
from transformers import (
    BertTokenizerFast, 
    BertForSequenceClassification, 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoConfig,
    BertForSequenceClassification, 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoConfig,
)
from convxai.writing_models.utils import *
from convxai.writing_models.trainers.diversity_trainer.data_loader import *


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')


class DiversityModel(object):

    def __init__(self, saved_model_dir = None):
        super(DiversityModel, self).__init__()

        if saved_model_dir is not None:
            config = AutoConfig.from_pretrained(saved_model_dir)
            config.num_labels = 5
            self.model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir, config=config).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
        else:
            config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
            config.num_labels = 5
            self.model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', config=config).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    
    def inference(self, inputs):
        """Generate Predictions for Users' Writing Inputs.
        """

        ### convert inputs into fragments
        inputs = paragraph_segmentation(inputs)

        ### convert inputs into dataloader 
        eval_dataloader = self.load_infer_data(inputs)

        ### make predictions
        pred_outs = self.generate_prediction(eval_dataloader)

        outputs = {
            "inputs": inputs,
            "outputs": pred_outs # pred_outs = (predict, probability)
        }

        return outputs


    def load_infer_data(self, x_text, batch_size = 32):
        feature = Feature(tokenizer=self.tokenizer)
        x_text = feature.extract(x_text[:])
        test_dataset = PredDataset(x_text)
        test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_dataloader


    def generate_prediction(self, eval_data):
        self.model.eval()
        predict = []
        probability = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():

            for _, (x_batch) in enumerate(eval_data, 1): # why start from 1?
                x_batch = x_batch.to(device)
                y_batch = torch.tensor([1]*x_batch.size(0)).to(device)
                outputs = self.model(x_batch, labels=y_batch)

                loss, y_pred = outputs[0:2]
                prob = torch.max(softmax(y_pred), dim=1)[0]
                y_pred = torch.argmax(y_pred, dim=1)

                predict.append(y_pred.cpu().numpy())
                probability.append(prob.cpu().numpy())

        predict = np.hstack(predict)
        probability = np.hstack(probability)

        return (predict, probability)


    def generate_confidence(self, inputs):

        ### convert inputs into fragments
        inputs = paragraph_segmentation(inputs)
        ### convert inputs into dataloader 
        eval_data = self.load_infer_data(inputs)

        self.model.eval()
        predict = []
        probability = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for _, (x_batch) in enumerate(eval_data, 1): # why start from 1?
                x_batch = x_batch.to(device)
                y_batch = torch.tensor([1]*x_batch.size(0)).to(device)
                outputs = self.model(x_batch, labels=y_batch)
                loss, y_pred = outputs[0:2]
                prob = torch.max(softmax(y_pred), dim=1)[0]
                y_pred = torch.argmax(y_pred, dim=1)
                predict.append(y_pred.cpu().numpy())
                probability.append(prob.cpu().numpy())
        predict = np.hstack(predict)
        probability = np.hstack(probability)
        return (predict, probability)


    def generate_embeddings(self,inputs):

        ### convert inputs into fragments
        inputs = paragraph_segmentation(inputs)
        
        ### convert inputs into dataloader 
        eval_data = self.load_infer_data(inputs)

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for _, (x_batch) in enumerate(eval_data, 1):
                x_batch = x_batch.to(device)
                y_batch = torch.tensor([1]*x_batch.size(0)).to(device)
                outputs = self.model(x_batch, labels=y_batch, output_hidden_states=True)
                hidden_states = outputs[2][-1]   ### hidden_states shape = torch.Size([7, 102, 768])
                embeddings.append(hidden_states.detach().cpu().numpy())
        embeddings = np.sum(np.concatenate(embeddings), axis=1)    # x_train_embeddings = (674, 768)
        return embeddings  ### shape = (7, 102, 768)



    # def generate_embeddings(self,inputs):

    #     ### convert inputs into fragments
    #     inputs = paragraph_segmentation(inputs)
        
    #     ### convert inputs into dataloader 
    #     eval_data = self.load_infer_data(inputs)

    #     self.model.eval()
    #     embeddings = []
    #     with torch.no_grad():
    #         for _, (x_batch) in enumerate(eval_data, 1):
    #             x_batch = x_batch.to(device)
    #             y_batch = torch.tensor([1]*x_batch.size(0)).to(device)
    #             outputs = self.model(x_batch, labels=y_batch, output_hidden_states=True)
    #             hidden_states = outputs[2][-1]   ### hidden_states shape = torch.Size([7, 102, 768])
    #             embeddings.append(hidden_states.detach().cpu().numpy())
    #     embeddings = np.sum(np.hstack(embeddings), axis=1)
    #     return (embeddings)  ### shape = (7, 102, 768)







    # def __init__(self, mode="inference") -> None:
    #     with open(os.path.join("../configs/diversity_model_config.json"), 'r') as fp:
    #         configs = json.load(fp)
    #     self.saved_model_dir = os.path.join(configs["save_dirs"]["root_dir"], configs["save_dirs"]["model_dir"])
    #     self.tokenizer, self.model = self.load_model(configs["model_version"])
    #     if mode == "inference":
    #         print(f"=== Loading Writing Model (Diversity Model) from: {self.saved_model_dir}")
    #         self.tokenizer.from_pretrained(self.saved_model_dir)
    #         self.model.load_state_dict(torch.load(os.path.join(self.saved_model_dir, "pytorch_model.bin"),  map_location=device))


    # def load_model(self, version):
    #     if version == "bert":
    #         print("Using Bert!!!")
    #         model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5).to(device)
    #         tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    #     elif version == "sci-bert":
    #         print("Using SCI-Bert!!!")
    #         config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
    #         config.num_labels = 5
    #         model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', config=config).to(device)
    #         tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    #     else:
    #         print("Loading Model Error: there is no model loaded!")
    #     tokenizer.save_pretrained(self.saved_model_dir)
    #     return tokenizer, model




    # def __init__(self) -> None:
    #     pass

    # def load_model(self, version, saved_model_dir = None):
    #     if version == "bert":
    #         if saved_model_dir is not None:
    #             self.model = BertForSequenceClassification.from_pretrained(saved_model_dir, num_labels=5).to(device)
    #             self.tokenizer = BertTokenizerFast.from_pretrained(saved_model_dir)
    #         else:
    #             print("Using Bert!!!")
    #             self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5).to(device)
    #             self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    #     elif version == "sci-bert":
    #         print("Using SCI-Bert!!!")
    #         if saved_model_dir is not None:
    #             config = AutoConfig.from_pretrained(saved_model_dir)
    #             config.num_labels = 5
    #             self.model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir, config=config).to(device)
    #             self.tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
    #         else:
    #             config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
    #             config.num_labels = 5
    #             self.model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', config=config).to(device)
    #             self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    #     else:
    #         print("Loading Model Error: there is no model loaded!")
    #     return self.tokenizer, self.model
