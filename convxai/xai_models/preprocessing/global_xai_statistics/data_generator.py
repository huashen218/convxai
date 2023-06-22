import os
import json


################## ICLR Conference ##################


import sys
sys.path.append('/data/hua/workspace/projects/convxai_user_study/src/')
from convxai.writing_models.models import *
from convxai.writing_models.utils import *
import stanza
import csv
import pandas as pd
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

cia_data_root_path = "/home/hqs5468/hua/workspace/projects/convxai_user_study/src/convxai/writing_models/data/cia"

conference = ["ACL", "CHI", "ICLR"]


diversity_model_label_mapping = {
    0: "background",
    1: "purpose",
    2: "method", 
    3: "finding",
    4: "other",
}



nlp = stanza.Pipeline('en')


def counting_tokens(abs_text):
    ### Extract Abstracts ###
    abs = abs_text.replace("\n", " ")
    doc = nlp(abs)

    ### Seg Sentences ###
    # sentences = [sentence.text for sentence in doc.sentences]
    tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
    return tokens



def main():
    
    ### Diversity_Model ###
    root_dir = "/data/hua/workspace/projects/convxai/src/convxai/"

    with open(os.path.join(root_dir, "writing_models/configs/diversity_model_config.json" ), 'r') as fp:
        diversity_model_config = json.load(fp)
    diversity_model = DiversityModel(diversity_model_config["save_dirs"]["output_dir"])

    ### Quality_Model ###
    with open(os.path.join(root_dir, "writing_models/configs/quality_model_config.json" ), 'r') as fp:
        quality_model_config = json.load(fp)
    quality_model = QualityModel(quality_model_config["save_configs"]["output_dir"])



    for conf in conference:
        result_dict = {}
        # token_no = 0
        # sentence_no = 0

        id_list = []
        abs_list = []
        aspect_list = []
        aspect_confidence_list = []
        perplexity = []
        token_counts = []

        filenames = [f for f in os.listdir(os.path.join(cia_data_root_path, conf)) if ".txt" in f]

        for id, f in tqdm(enumerate(filenames)):

            abs_dir = os.path.join(cia_data_root_path, conf, f)
            with open(abs_dir) as f:
                lines = f.readlines()
                abstract = paragraph_segmentation("".join(lines))
                

                # sentence_no += len(abstract)
                # token_no += len(tokens)

                #### for Diversity Model ####
                eval_dataloader = diversity_model.load_infer_data(abstract)
                diversity_pred_outs = diversity_model.generate_prediction(eval_dataloader)   #### Diversity Model output: (array([2, 2, 3, 0, 3, 3, 3, 3]), array([0.5996034 , 0.80663425, 0.42144373, 0.7498738 , 0.46669975, 0.6653462 , 0.8358171 , 0.9521568 ], dtype=float32))
                # print("Diversity Model output:", diversity_pred_outs)

                ### make predictions
                quality_pred_outs = quality_model.generate_prediction(abstract)  #### Diversity Model output: [46.5618792974957, 27.859385048005, 49.71423180886754, 192.63707746288597, 93.02696209629114, 62.89754600665361, 29.656592320962734, 87.89851739713316]
                # print("Diversity Model output:", quality_pred_outs)
                for i, abs in enumerate(abstract):
                    tokens = counting_tokens(abs)
                    id_list.append(id)
                    abs_list.append(abs)
                    aspect_list.append(diversity_pred_outs[0][i])
                    aspect_confidence_list.append(diversity_pred_outs[1][i])
                    perplexity.append(quality_pred_outs[i])
                    token_counts.append(len(tokens))

        result_dict = {
            "id": id_list,
            "text": abs_list,
            "aspect": aspect_list,
            "aspect_confidence": aspect_confidence_list,
            "perplexity": perplexity, 
            "token_count": token_counts,
        }


        save_dir = f"/data/hua/workspace/projects/convxai_user_study/src/convxai/xai_models/data/{conf}.csv"
        df = pd.DataFrame({ key:pd.Series(value) for key, value in result_dict.items() })
        df.to_csv(save_dir, index=False)


### Debug
if __name__ == '__main__':
    main()
