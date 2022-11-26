import json
import h5py
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel

from convxai.xai_models.utils import h5_load


def nlu_f2q_to_q2f(nlu_form2question_dir, nlu_question2form_dir):

    with open(nlu_form2question_dir) as f:
        data = json.load(f)

    questions, forms = [], []
    for i, (keys, values) in enumerate(data.items()):
        for v in values:
            forms.append(keys)
            questions.append(v)
    df = pd.DataFrame({'questions': questions,
                        'forms': forms})
    df.to_csv(nlu_question2form_dir)



def save_question_embeddings(nlu_question2form_dir, nlu_question_embedding_dir):

    data = pd.read_csv(nlu_question2form_dir)
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")

    question_embeddings = []
    labels = []
    for q, f in zip(data['questions'],data['forms']):
        inputs = tokenizer(q, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state[0][0]
        question_embeddings.append(last_hidden_states.cpu().detach().numpy())
        labels.append(f)

    with h5py.File(nlu_question_embedding_dir, "w") as outfile:
        outfile.create_dataset("question_embeddings", data=question_embeddings)
        outfile.create_dataset("labels", data=labels)





def main():
    nlu_form2question_dir = "./nlu_form2question.json"
    nlu_question2form_dir = "./nlu_question2form.csv"
    nlu_question_embedding_dir = "./nlu_question_embedding.h5"

    ####### Data Preprocessing #######
    nlu_f2q_to_q2f(nlu_form2question_dir, nlu_question2form_dir)

    ####### Train Sci-BERT Model and Save Embeddings ######
    save_question_embeddings(nlu_question2form_dir, nlu_question_embedding_dir)

    question_embeddings, labels  = h5_load(nlu_question_embedding_dir, [
                                    "question_embeddings", "labels"],  verbose=True)
    print("question_embeddings:", question_embeddings)
    print('labels', labels)



if __name__ == "__main__":
    main()


