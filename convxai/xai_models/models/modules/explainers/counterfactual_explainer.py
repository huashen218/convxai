import torch
import openai
import os
import json
import re
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from typing import Optional
from pprint import pprint
from sentence_transformers import SentenceTransformer, util
from convxai.utils import parse_system_config_file

label_mapping = {
    0: "background",
    1: "purpose",
    2: "method", 
    3: "finding",
    4: "other",
}

class GPT3():
    def __init__(self):
        openai.organization = "org-TOv0UMKmeZOrNkJwMOJmN63g"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.space_pattern = re.compile(r"\s+")
        self.parameters = {
            "model": "text-davinci-003",
            # "model": "text-curie-001",
            # "model": "text-ada-001",
            "temperature": 0.95,
            "max_tokens": 100,
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        }

    def generate(self, prompt:str):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            prompt=prompt,
            **self.parameters,
        )
        return response

class CounterfactualExplainer():
    def __init__(
        self,
        sentence_model_name="all-MiniLM-L6-v2",
    ):
        # settings
        self.system_config = parse_system_config_file()
        data_folder = os.path.join(self.system_config['conversational_xai']['checkpoints_root_dir'], self.system_config['conversational_xai']['xai_counterfactual_dir'])

        # load data
        self.embeddings = joblib.load(Path(data_folder, f"embeddings_{sentence_model_name}.joblib"))
        self.text_info = self.load_json(Path(data_folder, "text_info.json"))
        self.text_info = pd.DataFrame(self.text_info)

        # build sentence-transformer
        self.model = SentenceTransformer(sentence_model_name)

        # build gpt3
        self.gpt3 = GPT3()

    def load_json(self, path:Path):
        with open(path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        return data

    def build_prompt(
        self, 
        input_text:str,
        contrast_label:int,
        num_example:Optional[int] = 5,
        conference:Optional[str] = None,
    ) -> str:
        input_embedding = self.model.encode(input_text)
        cosine_scores = util.cos_sim(input_embedding, self.embeddings)[0]

        # filtering (Part of this can be built offline)
        sample_list = []        
        for aspect_index in np.arange(0, 5):
            if conference is None:
                filtering = self.text_info["aspect"] == aspect_index
                index_list = self.text_info.index[filtering]
                sub_text_info = self.text_info[filtering]
            else: 
                filtering = self.text_info["aspect"] == aspect_index
                filtering = filtering * (self.text_info["conference"] == conference)
                index_list = self.text_info.index[filtering]
                sub_text_info = self.text_info[filtering]

            # select by cosine similarity            
            best_index_list = torch.argsort(
                cosine_scores[index_list],
                descending=True,
            )[:num_example]

            for best_index in best_index_list:
                sample = sub_text_info.iloc[int(best_index)]
                sample_list.append({
                    "aspect": aspect_index,
                    "text": sample["text"],
                    "conference": sample["conference"],
                })

        prompt_example = "\n".join([
            f"\"{sample['text']}\" is labeled \"{label_mapping[sample['aspect']]}\""
            for sample in sample_list
        ])

        prompt = (
            f"Given the following examples:\n{prompt_example}\n\n\n"
            f"Rewrite \"{input_text}\" into label \"{label_mapping[contrast_label]}\": "
        )
        return prompt



    def generate_counterfactual(
        self, 
        input_text:str, 
        contrast_label:int,
        conference:Optional[str] = None
    ):
        # contrast_label = (contrast_label + len(label_mapping)) % len(label_mapping)

        prompt = self.build_prompt(
            input_text, 
            contrast_label=contrast_label, 
            conference=conference,
        )

        response = self.gpt3.generate(prompt)
        response["prompt"] = prompt

        # with open("gpt3-text-3.json", 'w', encoding='utf-8') as outfile:
        #     json.dump(response, outfile, indent=2)

        return {
            "original_input": input_text,
            "counterfactual_input": response["choices"][0]["text"].strip(),
            "counterfactual_label": label_mapping[contrast_label],
            "counterfactual_confidence": 0.0,
        }

def main():
    counter_factual = CounterfactualExplainer()

    counter_factual.generate_counterfactual(
        input_text = "However, a number of problems of recent interest have created a demand for models that can analyze spherical images.",
        contrast_label = 1,
        conference = "ICLR",
    )

if __name__ == "__main__":
    main()