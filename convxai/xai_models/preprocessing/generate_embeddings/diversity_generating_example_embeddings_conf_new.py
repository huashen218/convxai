import copy
import json
import h5py
import torch
import argparse

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import torch
import numpy as np
from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from convxai.writing_models.utils import *
from convxai.writing_models.models import *
from convxai.writing_models.trainers.diversity_trainer.trainer import Trainer
from convxai.writing_models.trainers.diversity_trainer.data_loader import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






nlp = stanza.Pipeline('en')


def counting_tokens(abs_text):
    ### Extract Abstracts ###
    abs = abs_text.replace("\n", " ")
    try:
        doc = nlp(abs)
        ### Seg Sentences ###
        # sentences = [sentence.text for sentence in doc.sentences]
        tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
        return tokens
    except:
        return []





def generate_embeddings(inputs, diversity_model):

    ### convert inputs into fragments
    # inputs = paragraph_segmentation(inputs)
    
    ### convert inputs into dataloader 
    eval_data = diversity_model.load_infer_data(inputs)

    diversity_model.model.eval()
    embeddings = []
    with torch.no_grad():
        for _, (x_batch) in tqdm(enumerate(eval_data, 1)):
            x_batch = x_batch.to(device)
            y_batch = torch.tensor([1]*x_batch.size(0)).to(device)
            outputs = diversity_model.model(x_batch, labels=y_batch, output_hidden_states=True)
            hidden_states = outputs[2][-1]   ### hidden_states shape = torch.Size([7, 102, 768])
            embeddings.append(hidden_states.detach().cpu().numpy())   ### embeddings shape = (674, 102, 768)
    embeddings = np.sum(np.concatenate(embeddings), axis=1)    # x_train_embeddings = (674, 768)
    return embeddings  ### shape = (sent_count, embedding)





def save_embeddings(texts, titles, links, diversity_model, conference, aspect_list, aspect_confidence_list, perplexity, token_counts, abstract_seg_scores):

    # x_train_embeddings = generate_embeddings(texts, diversity_model)

    # embedding_save_dir = f"/home/hqs5468/hua/workspace/projects/convxai/src/convxai/xai_models/data/explainer/xai_example_data/embeddings/diversity_model_{conference}_embeddings.h5"
    text_save_dir      = f"/home/hqs5468/hua/workspace/projects/convxai/src/convxai/xai_models/data/explainer/xai_example_data/embeddings/diversity_model_{conference}_texts.h5"

    # ### Save embeddings ###
    # with h5py.File(embedding_save_dir, 'w') as outfile:
    #     outfile.create_dataset("x_train_embeddings", data=x_train_embeddings) 

    # if os.path.isfile(embedding_save_dir):
    #     with h5py.File(embedding_save_dir, 'r') as infile:
    #         x_train_embeddings_tmp = np.empty(infile["x_train_embeddings"].shape, dtype=infile["x_train_embeddings"].dtype)
    #         infile["x_train_embeddings"].read_direct(x_train_embeddings_tmp)


    ### Save texts ###
    with h5py.File(text_save_dir, 'w') as outfile:
        outfile.create_dataset("x_train_text", data=texts) 
        outfile.create_dataset("x_train_title", data=titles) 
        outfile.create_dataset("x_train_link", data=links) 

        outfile.create_dataset("aspect_list", data=aspect_list) 
        outfile.create_dataset("aspect_confidence_list", data=aspect_confidence_list) 
        outfile.create_dataset("perplexity", data=perplexity) 
        outfile.create_dataset("token_counts", data=token_counts) 
        outfile.create_dataset("abstract_seg_scores", data=abstract_seg_scores) 



    if os.path.isfile(text_save_dir):
        with h5py.File(text_save_dir, 'r') as infile:
            x_train_text_tmp = np.empty(infile["x_train_text"].shape, dtype=infile["x_train_text"].dtype)
            infile["x_train_text"].read_direct(x_train_text_tmp)
            x_train_title_tmp = np.empty(infile["x_train_title"].shape, dtype=infile["x_train_title"].dtype)
            infile["x_train_title"].read_direct(x_train_title_tmp)
            x_train_link_tmp = np.empty(infile["x_train_link"].shape, dtype=infile["x_train_link"].dtype)
            infile["x_train_link"].read_direct(x_train_link_tmp)


            aspect_list_tmp = np.empty(infile["aspect_list"].shape, dtype=infile["aspect_list"].dtype)
            infile["aspect_list"].read_direct(aspect_list_tmp)

            aspect_confidence_list_tmp = np.empty(infile["aspect_confidence_list"].shape, dtype=infile["aspect_confidence_list"].dtype)
            infile["aspect_confidence_list"].read_direct(aspect_confidence_list_tmp)

            perplexity_tmp = np.empty(infile["perplexity"].shape, dtype=infile["perplexity"].dtype)
            infile["perplexity"].read_direct(perplexity_tmp)

            token_counts_tmp = np.empty(infile["token_counts"].shape, dtype=infile["token_counts"].dtype)
            infile["token_counts"].read_direct(token_counts_tmp)

            abstract_seg_scores_tmp = np.empty(infile["abstract_seg_scores"].shape, dtype=infile["abstract_seg_scores"].dtype)
            infile["abstract_seg_scores"].read_direct(abstract_seg_scores_tmp)


# ===== From CHI.json to embedding.h5 ======
def main(args):

    ############# Load Model ############
    with open(os.path.join(args.config_dir), 'r') as fp:
        configs = json.load(fp)
    model_dir = os.path.join(configs["save_dirs"]["root_dir"], configs["save_dirs"]["model_dir"])
    diversity_model  = DiversityModel(model_dir)
    diversity_model.model.to(device)



    ### Quality_Model ###
    root_dir = "/data/hua/workspace/projects/convxai/src/convxai/"
    with open(os.path.join(root_dir, "writing_models/configs/quality_model_config.json" ), 'r') as fp:
        quality_model_config = json.load(fp)
    quality_model = QualityModel(quality_model_config["save_configs"]["output_dir"])
    quality_model.model.to(device)



    ############# Load Json File ############
    crawl_data_dir = "/home/hqs5468/hua/workspace/projects/convxai/src/convxai/writing_models/data/cia/crawl_data/"






#    ############# ACL ############
#     conference = "ACL"

#     data_save_dir = os.path.join(crawl_data_dir, f"{conference}_papers.json")
#     with open(data_save_dir, 'r') as f:
#         data = json.load(f)
#     print(f"====>>>>>>> crawling data dir =  {data_save_dir}")


#     texts = []
#     titles = []
#     links = []

#     aspect_list = []
#     aspect_confidence_list = []
#     perplexity = []
#     token_counts = []
#     abstract_seg_scores = []

#     for k in tqdm(range(len(data))):
#         abstract = data[k]['abstract']
#         title    = data[k]['title']
#         link     = data[k]['url']

#         abstract_seg = abstract.split('. ')
#         texts.extend(abstract_seg)
#         titles.extend([title] * len(abstract_seg))
#         links.extend([link] * len(abstract_seg))

#         #### for Diversity Model ####
#         eval_dataloader = diversity_model.load_infer_data(abstract_seg)
#         diversity_pred_outs = diversity_model.generate_prediction(eval_dataloader)   #### Diversity Model output: (array([2, 2, 3, 0, 3, 3, 3, 3]), array([0.5996034 , 0.80663425, 0.42144373, 0.7498738 , 0.46669975, 0.6653462 , 0.8358171 , 0.9521568 ], dtype=float32))

#         ### make predictions
#         quality_pred_outs = quality_model.generate_prediction(abstract_seg)  #### Diversity Model output: [46.5618792974957, 27.859385048005, 49.71423180886754, 192.63707746288597, 93.02696209629114, 62.89754600665361, 29.656592320962734, 87.89851739713316]

#         abstract_seg_score = []
#         for i, abs in enumerate(abstract_seg):
#             abstract_seg_score.append(quality_pred_outs[i])
#             aspect_list.extend([diversity_pred_outs[0][i]])
#             aspect_confidence_list.extend([diversity_pred_outs[1][i]])
#             perplexity.extend([quality_pred_outs[i]])
#             tokens = counting_tokens(abs)
#             token_counts.extend([len(tokens)])
#         abstract_seg_scores.extend([np.mean(abstract_seg_score)])

#     save_embeddings(texts, titles, links, diversity_model, conference, aspect_list, aspect_confidence_list, perplexity, token_counts, abstract_seg_scores)

    



# #    ############# ICLR ############
#     conference = "ICLR"

#     data_save_dir = os.path.join(crawl_data_dir, f"{conference}_papers.json")
#     with open(data_save_dir, 'r') as f:
#         data = json.load(f)
#     print(f"====>>>>>>> crawling data dir =  {data_save_dir}")

#     texts = []
#     titles = []
#     links = []

#     aspect_list = []
#     aspect_confidence_list = []
#     perplexity = []
#     token_counts = []
#     abstract_seg_scores = []


#     for k in tqdm(range(len(data))):
#         abstract = data[k]['Abstract']
#         title    = data[k]['title']
#         link     = "https://openreview.net" + data[k]['url']


#         abstract_seg = abstract.split('. ')
#         texts.extend(abstract_seg)
#         titles.extend([title] * len(abstract_seg))
#         links.extend([link] * len(abstract_seg))



#         #### for Diversity Model ####
#         eval_dataloader = diversity_model.load_infer_data(abstract_seg)
#         diversity_pred_outs = diversity_model.generate_prediction(eval_dataloader)   #### Diversity Model output: (array([2, 2, 3, 0, 3, 3, 3, 3]), array([0.5996034 , 0.80663425, 0.42144373, 0.7498738 , 0.46669975, 0.6653462 , 0.8358171 , 0.9521568 ], dtype=float32))


#         ### make predictions
#         quality_pred_outs = quality_model.generate_prediction(abstract_seg)  #### Diversity Model output: [46.5618792974957, 27.859385048005, 49.71423180886754, 192.63707746288597, 93.02696209629114, 62.89754600665361, 29.656592320962734, 87.89851739713316]

#         abstract_seg_score = []
#         for i, abs in enumerate(abstract_seg):
#             abstract_seg_score.append(quality_pred_outs[i])
#             aspect_list.extend([diversity_pred_outs[0][i]])
#             aspect_confidence_list.extend([diversity_pred_outs[1][i]])
#             perplexity.extend([quality_pred_outs[i]])
#             tokens = counting_tokens(abs)
#             token_counts.extend([len(tokens)])
#         abstract_seg_scores.extend([np.mean(abstract_seg_score)])

#     save_embeddings(texts, titles, links, diversity_model, conference, aspect_list, aspect_confidence_list, perplexity, token_counts, abstract_seg_scores)

    





#    ############# CHI ############
    conference = "CHI"

    data_save_dir = os.path.join(crawl_data_dir, f"{conference}_papers.json")
    with open(data_save_dir, 'r') as f:
        data = json.load(f)
    print(f"====>>>>>>> crawling data dir =  {data_save_dir}")


    texts = []
    titles = []
    links = []

    aspect_list = []
    aspect_confidence_list = []
    perplexity = []
    token_counts = []
    abstract_seg_scores = []

    for k in tqdm(range(len(data))):
        abstract = data[k]['abstract']
        title    = data[k]['title']
        link     = data[k]['URL']
        # keyword = data[k]['keyword']

        abstract_seg = abstract.split('. ')
        texts.extend(abstract_seg)
        titles.extend([title] * len(abstract_seg))
        links.extend([link] * len(abstract_seg))

        #### for Diversity Model ####
        eval_dataloader = diversity_model.load_infer_data(abstract_seg)
        diversity_pred_outs = diversity_model.generate_prediction(eval_dataloader)   #### Diversity Model output: (array([2, 2, 3, 0, 3, 3, 3, 3]), array([0.5996034 , 0.80663425, 0.42144373, 0.7498738 , 0.46669975, 0.6653462 , 0.8358171 , 0.9521568 ], dtype=float32))

        ### make predictions
        quality_pred_outs = quality_model.generate_prediction(abstract_seg)  #### Diversity Model output: [46.5618792974957, 27.859385048005, 49.71423180886754, 192.63707746288597, 93.02696209629114, 62.89754600665361, 29.656592320962734, 87.89851739713316]


        abstract_seg_score = []
        for i, abs in enumerate(abstract_seg):
            abstract_seg_score.append(quality_pred_outs[i])
            aspect_list.extend([diversity_pred_outs[0][i]])
            aspect_confidence_list.extend([diversity_pred_outs[1][i]])
            perplexity.extend([quality_pred_outs[i]])
            tokens = counting_tokens(abs)
            token_counts.extend([len(tokens)])
        abstract_seg_scores.extend([np.mean(abstract_seg_score)])

    save_embeddings(texts, titles, links, diversity_model, conference, aspect_list, aspect_confidence_list, perplexity, token_counts, abstract_seg_scores)

    






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diversity Model for Scientific Writing Support.")
    parser.add_argument("--config_dir", dest="config_dir", help="config file path", type=str, default=".../configs/diversity_model_config.json")
    parser.add_argument("--data_save_dir", dest="data_save_dir", type=str)
    args = parser.parse_args()
    main(args)