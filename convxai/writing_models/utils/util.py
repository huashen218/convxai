import os
import torch
import stanza
import pandas as pd
import numpy as np





diversity_model_label_mapping = {
    0: "background",
    1: "purpose",
    2: "method", 
    3: "finding",
    4: "other",
}


nlp = stanza.Pipeline('en')
# nlp = stanza.Pipeline('en', use_gpu=False)



def counting_tokens(abs_text):
    ### Extract Abstracts ###
    abs = abs_text.replace("\n", " ")
    doc = nlp(abs)

    ### Seg Sentences ###
    # sentences = [sentence.text for sentence in doc.sentences]
    tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
    return tokens



# splitPuctList = [",", ";", "."]
splitPuctList = [";", "."]
LB = ["(", "[", "{"] 
RB = [")", "]", "}"]
minTokenNum = 6


def paragraph_segmentation(abs_text):

    # nlp = stanza.Pipeline('en')

    ### Extract Abstracts ###
    abs = abs_text.replace("\n", " ")
    doc = nlp(abs)

    ### Seg Sentences ###
    # sentences = [sentence.text for sentence in doc.sentences]
    tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]

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

    # original case
    # text_fragments = [re.split('; |, |\*', sentence) for sentence in sentences]

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
    # return sentences_content
            fragment_content.append(text_fragments[sent_id][frag_id])
    return fragment_content



def create_folder(folder_list):   #[model_dir, log_dir, result_dir, cache_dir]
    for folder in folder_list:
        if not os.path.isdir(folder):
            os.makedirs(folder)
    print("Created folders:", folder_list)










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



def output_score(score):
    table = pd.DataFrame(
        [score[3], score[0], score[1], score[2]],
        index=["# samples", "Precision", "Recall", "F1"],
        columns=["background", "purpose", "method", "finding", "other"],
    )
    return table







