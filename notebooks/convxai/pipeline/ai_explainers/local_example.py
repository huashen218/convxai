
import h5py
import torch.nn as nn
import numpy as np
from convxai.utils import *

from ..ai_models import *


class ExampleExplainer(object):
    
    def __init__(self, conference):
        super().__init__()

        self.system_config = parse_system_config_file()
        self.model_configs = self.system_config['conversational_xai']['checkpoints_root_dir']

        self.model_embeddings_h5dir = os.path.join(self.model_configs, self.system_config['conversational_xai']['xai_example_dir']["xai_emample_embeddings_dir"][conference])
        self.model_texts_h5dir = os.path.join(self.model_configs, self.system_config['conversational_xai']['xai_example_dir']["xai_emample_texts_dir"][conference])

        with h5py.File(self.model_embeddings_h5dir, 'r') as infile:
            self.x_train_embeddings_tmp = np.empty(infile["x_train_embeddings"].shape, dtype=infile["x_train_embeddings"].dtype)
            infile["x_train_embeddings"].read_direct(self.x_train_embeddings_tmp)

        with h5py.File(self.model_texts_h5dir, 'r') as infile:
            self.x_train_text_tmp = np.empty(infile["x_train_text"].shape, dtype=infile["x_train_text"].dtype)
            infile["x_train_text"].read_direct(self.x_train_text_tmp)
            self.x_train_title_tmp = np.empty(infile["x_train_title"].shape, dtype=infile["x_train_title"].dtype)
            infile["x_train_title"].read_direct(self.x_train_title_tmp)
            self.x_train_link_tmp = np.empty(infile["x_train_link"].shape, dtype=infile["x_train_link"].dtype)
            infile["x_train_link"].read_direct(self.x_train_link_tmp)

            self.aspect_list_tmp = np.empty(infile["aspect_list"].shape, dtype=infile["aspect_list"].dtype)
            infile["aspect_list"].read_direct(self.aspect_list_tmp)
            self.aspect_confidence_list_tmp = np.empty(infile["aspect_confidence_list"].shape, dtype=infile["aspect_confidence_list"].dtype)
            infile["aspect_confidence_list"].read_direct(self.aspect_confidence_list_tmp)
            self.perplexity_tmp = np.empty(infile["perplexity"].shape, dtype=infile["perplexity"].dtype)
            infile["perplexity"].read_direct(self.perplexity_tmp)
            self.token_counts_tmp = np.empty(infile["token_counts"].shape, dtype=infile["token_counts"].dtype)
            infile["token_counts"].read_direct(self.token_counts_tmp)



def explain_example(model, input, predict_label, conference, **kwargs):
    """XAI Algorithm - Confidence: 
        Paper: An Empirical Comparison of Instance Attribution Methods for NLP (https://aclanthology.org/2021.naacl-main.75.pdf)
    """

    ######### User Input Variable #########
    top_k = kwargs["attributes"]["top_k"] if kwargs["attributes"]["top_k"] is not None else 3
    label = label_mapping[kwargs["attributes"]["aspect"]] if kwargs["attributes"]["aspect"] is not None else predict_label
    keyword = kwargs["attributes"]["keyword"] if kwargs["attributes"]["keyword"] is not None else None
    rank    = kwargs["attributes"]["rank"] if kwargs["attributes"]["rank"] is not None else None

    ######### Explaining #########
    example_explainer = ExampleExplainer(conference)
    embeddings = model.generate_embeddings(input)
    filter_index = np.where(example_explainer.aspect_list_tmp == label)[0]
    find_examples = True

    if keyword is not None:
        keyword_filter_index = []
        for idx, text in enumerate(example_explainer.x_train_text_tmp):
            if keyword in text.decode("utf-8"):
                keyword_filter_index.append(idx)
        if len(keyword_filter_index)>0:
            filter_index = np.array(list(set(keyword_filter_index).intersection(set(list(filter_index)))))
        else:
            find_examples = False

    if rank is not None:
        rank_filter_index = []
        rank_top = int(len(example_explainer.token_counts_tmp) * 0.5)
        if rank == 'quality_score':
            rank_filter_index = np.argsort(example_explainer.perplexity_tmp)[:rank_top]
        elif rank == 'short':
            rank_filter_index = np.argsort(example_explainer.token_counts_tmp)[:rank_top]
        elif rank == 'long':
            rank_filter_index = np.argsort(example_explainer.token_counts_tmp)[::-1][:rank_top]

        if len(rank_filter_index)>0:
            filter_index = np.array(list(set(rank_filter_index).intersection(set(list(filter_index)))))
        else:
            find_examples = False

    example_explainer.x_train_embeddings_tmp = np.array(example_explainer.x_train_embeddings_tmp)[filter_index]
    similarity_scores = np.matmul(embeddings, np.transpose(example_explainer.x_train_embeddings_tmp, (1,0)))[0]    ###### train_emb = torch.Size(137171, 768)  
    final_top_index = filter_index[np.argsort(similarity_scores)[::-1][:top_k]]

    top_text = np.array(example_explainer.x_train_text_tmp)
    top_link = np.array(example_explainer.x_train_link_tmp)

    if find_examples:
        nlg_template = f"The top-{top_k} similar examples (i.e., of selected-sentence = '<i><span class='text-info'>{input}</span>') from the <strong>{conference}</strong> dataset are (<strong>label={diversity_model_label_mapping[label]}, rank={rank}</strong>):"
        for t in final_top_index:
            nlg_template += f"<br> <strong>sample-{t+1}</strong> - <a class='post-link' href='{top_link[t].decode('UTF-8')}' target='_blank'>{top_text[t].decode('UTF-8')}</a>."
    else:
        nlg_template = f"We didn't find <strong>keyword={keyword}</strong>, <strong>label={diversity_model_label_mapping[label]}</strong>, and <strong>rank={rank}</strong> in the dataset. Please try with other keywords or labels to find the similar examples again."

    response = nlg_template
    return response


