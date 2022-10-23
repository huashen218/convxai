import os
import yaml
import h5py
import stanza
import logging
import numpy as np
from pymongo import MongoClient


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def create_folder(folder_list):
    """Create the folders if the pathes do not exist.
    """
    for folder in folder_list:
        if not os.path.isdir(folder):
            os.makedirs(folder)
        logging.info(f"Created folders at {folder_list}")



def h5_load(filename, data_list, dtype=None, verbose=False):
    """Load h5 files.
    """

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
            logging.info("\n".join(
                "{} = {} [{}]".format(data_name, str(real_data.shape), str(real_data.dtype))
                for data_name, real_data in zip(data_list, data)
            ))
            logging.info()
        return data




nlp = stanza.Pipeline('en')

splitPuctList = [";", "."]
LB = ["(", "[", "{"] 
RB = [")", "]", "}"]
minTokenNum = 6

def paragraph_segmentation(abs_text):

    # nlp = stanza.Pipeline('en')

    ### Extract Abstracts ###
    abs = abs_text.replace("\n", " ")
    doc = nlp(abs)

    #### Segmentation Rules ####
    ## 1 - We don't break brackets (i.e., (), {}, [])
    ## 2 - We use commas (,), semicolons (;), periods (.) to split each sentence.
    ## 3 - For fragments fewer than six tokens, we don't touch;
    text_fragments = []
    for sentence in doc.sentences:
        bracket_break = False
        fragments = ""
        token_count = 0

        sent_segment = []

        for tok in sentence.tokens:
            token = tok.text
            fragments += token+" "
            token_count += 1

            if token in LB:
                bracket_break = True
            if token in RB:
                bracket_break = False 

            if (token == ".") or (token in splitPuctList and token_count > minTokenNum and not bracket_break):
                    sent_segment.append(fragments.strip().replace("- ", "-").replace(" -", "-"))
                    fragments = ""
                    token_count = 0  ## NEW Added.
            else:
                pass
        text_fragments.append(sent_segment)

    segment_num = 0
    sentences_content = []
    fragment_content = []

    for sent_id in range(len(text_fragments)):
        sentences_content.append([])
        for frag_id in range(len(text_fragments[sent_id])):
            segment_num += 1
            frag_content = {}
            frag_content["segment_text"] = text_fragments[sent_id][frag_id]
            frag_content["crowd_label"] = ""
            sentences_content[sent_id].append(frag_content)
            fragment_content.append(text_fragments[sent_id][frag_id])
    return fragment_content




class EarlyStop:
    def __init__(self, mode="max", history=5):
        if mode == "max":
            self.best_func = np.max
            self.best_val = -np.inf 
            self.comp = lambda x, y: x >= y
        elif mode == "min":
            self.best_func = np.min
            self.best_val = np.inf 
            self.comp = lambda x, y: x <= y
        else:
            print("Please use 'max' or 'min' for mode.")
            quit()
        
        self.history_num = history
        self.history = np.zeros((self.history_num, ))
        self.total_num = 0

    def check(self, score):
        self.history[self.total_num % self.history_num] = score
        self.total_num += 1
        current_best_val = self.best_func(self.history)
        
        if self.total_num <= self.history_num:
            return False

        if self.comp(current_best_val, self.best_val):
            self.best_val = current_best_val
            return False
        else:
            return True










