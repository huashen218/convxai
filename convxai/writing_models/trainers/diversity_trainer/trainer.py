#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_model = None
best_accuracy = 0.0
best_epoch = 0

class Trainer(object):

    def __init__(self, configs):
        super(Trainer, self).__init__()
        self.configs = configs


    def train(self, epoch, model, dataloader, optimizer):
        predictions_labels = []
        true_labels = []
        total_loss = 0
        total_acc = 0
        model.train()

        for count, (x_batch, y_batch) in enumerate(dataloader, 1):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch, labels=y_batch)
            loss, y_pred = outputs[0:2]

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
            y_pred = torch.argmax(y_pred, dim=1)
            correct_num = torch.sum(y_pred == y_batch).double()
            total_acc += correct_num / y_pred.shape[0]
            predictions_labels.append(y_pred.cpu().numpy())
            true_labels.append(y_batch.cpu().numpy())

            print("\x1b[2K\rEpoch: {} / {} Loss: {:.5f} Acc: {:.5f}".format(
                epoch, self.configs["model_params"]["scibert_param"]["epoch_num"], total_loss/count, total_acc/count), end="")

        return 


    def evaluate(self, epoch, model, tokenizer, dataloader, model_dir=None):
        global best_accuracy, best_epoch
        best_accuracy = 0.0 if epoch == 0 else best_accuracy
        best_epoch = 0 if epoch == 0 else best_epoch

        predictions_labels = []
        true_labels = []
        total_loss = 0
        total_acc = 0
        model.eval()

        with torch.no_grad():
            for count, (x_batch, y_batch) in enumerate(dataloader, 1):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_batch, labels=y_batch)
                loss, y_pred = outputs[0:2]

                # compute loss
                total_loss += loss.item()

                # compute accuracy
                y_pred = torch.argmax(y_pred, dim=1)
                correct_num = torch.sum(y_pred == y_batch).double()
                total_acc += correct_num / y_pred.shape[0]
                predictions_labels.append(y_pred.cpu().numpy())
                true_labels.append(y_batch.cpu().numpy())

                print("\x1b[2K\rEval Loss: {:.5f} Acc: {:.5f}".format(total_loss/count, total_acc/count), end="")
            

        predictions_labels = np.hstack(predictions_labels)
        true_labels = np.hstack(true_labels)
        acc = accuracy_score(true_labels, predictions_labels)
        print("\nAccuracy: {:.5f}\n".format(acc))


        if model_dir is not None:

            if acc > best_accuracy:
                best_model = copy.deepcopy(model.state_dict())
                best_accuracy = acc
                best_epoch = epoch

                model.load_state_dict(best_model)
                print("saving best model at epoch {}".format(best_epoch))
                model.save_pretrained(self.saved_model_dir)
                tokenizer.save_pretrained(self.saved_model_dir)


        return acc, predictions_labels, true_labels


