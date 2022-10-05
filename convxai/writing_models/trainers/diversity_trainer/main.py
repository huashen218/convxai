import copy
import json
import h5py
import torch
import argparse

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from convxai.writing_models.utils import *
from convxai.writing_models.models import *
from convxai.writing_models.trainers.diversity_trainer.trainer import Trainer
from convxai.writing_models.trainers.diversity_trainer.data_loader import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    ############# Set Configs ############
    with open(os.path.join(args.config_dir), 'r') as fp:
        configs = json.load(fp)

    ############# Set Dirs ###########c#
    model_dir = os.path.join(configs["save_dirs"]["root_dir"], configs["save_dirs"]["model_dir"])
    cache_dir = os.path.join(configs["save_dirs"]["root_dir"], configs["save_dirs"]["cache_dir"])
    create_folder([model_dir, cache_dir])

    ############# Load Model ############
    diversity_model  = DiversityModel()


    ############ Load Dataset ############
    train_dataloader, dev_dataloader, test_dataloader = data_loader(configs, cache_dir, diversity_model.tokenizer)


    ############# Start Training ############
    learning_rate       = configs["model_params"]["scibert_param"]["learning_rate"]
    optimizer = torch.optim.Adam(diversity_model.model.parameters(), lr=learning_rate)
    early_stop_epoch    = configs["model_params"]["scibert_param"]["early_stop_epoch"]
    stopper = EarlyStop(mode="max", history=early_stop_epoch)
    trainer = Trainer(configs)


    print(f" ====== Mode = {args.mode} ====== ")
    if args.mode != "evaluate":
        print('\t-----------------------------------------------------------------------')
        print(f'\t ============= Diversity Model Training =============')
        print('\t-----------------------------------------------------------------------')
        for epoch in range(1, configs["model_params"]["scibert_param"]["epoch_num"]+1):
            print(f'\t ========================== Epoch: {epoch:02} ==========================')
            trainer.train(epoch, diversity_model.model, train_dataloader, optimizer)
            acc, _, _ = trainer.evaluate(epoch, diversity_model.model, diversity_model.tokenizer, dev_dataloader, model_dir)
            # check early stopping
            if stopper.check(acc):
                print("Early Stopping at Epoch = ", epoch)
                break


    ############# Start Evaluation #############
    print('\t-----------------------------------------------------------------------')
    print(f'\t ============= Diversity Model Evaluating =============')
    print('\t-----------------------------------------------------------------------')
    diversity_model  = DiversityModel(model_dir)


    acc, predict, true_label = trainer.evaluate(configs["model_params"]["scibert_param"]["epoch_num"], diversity_model.model, diversity_model.tokenizer, test_dataloader) 
    score = precision_recall_fscore_support(true_label, predict)
    table = output_score(score)
    print(table)

    with open(os.path.join(model_dir, "all_results.json"), "w") as f:
        json.dump({"accuracy": acc}, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diversity Model for Scientific Writing Support.")
    parser.add_argument("--config_dir", dest="config_dir", help="config file path", type=str, default=".../configs/diversity_model_config.json")
    parser.add_argument("--mode", dest="mode", help="train/evaluate", type=str, default="train")
    args = parser.parse_args()
    main(args)