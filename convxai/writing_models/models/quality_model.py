import os
import math
import torch
import logging
import torch.nn as nn
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from convxai.writing_models.utils import *
from convxai.writing_models.trainers.diversity_trainer.data_loader import *
from tqdm import tqdm

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')


class QualityModel(object):

    def __init__(self, saved_model_dir = "gpt2"):
        config = AutoConfig.from_pretrained(saved_model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(saved_model_dir, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            saved_model_dir,
            from_tf=bool(".ckpt" in saved_model_dir),
            config=config,
        ).to(device)


    def inference(self, inputs):
        """Generate Predictions for Users' Writing Inputs.
        """

        ### convert inputs into fragments
        inputs = paragraph_segmentation(inputs)

        ### make predictions
        pred_outs = self.generate_prediction(inputs)

        outputs = {
            "inputs": inputs,
            "outputs": pred_outs # pred_outs = (perplexities)
        }

        return outputs




    def generate_prediction(self, eval_data):
        self.model.eval()
        perplexities = []
        with torch.no_grad():
            for input_text in eval_data:
                inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
                inputs['labels'] = inputs['input_ids']
                outputs = self.model(**inputs)
                loss = outputs.loss
                perplexities.append(math.exp(loss))
        return (perplexities)



    def generate_confidence(self, inputs):

        ### convert inputs into fragments
        eval_data = paragraph_segmentation(inputs)

        self.model.eval()
        perplexities = []
        probability = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for input_text in eval_data:
                inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
                inputs['labels'] = inputs['input_ids']
                outputs = self.model(**inputs)
                loss, y_pred = outputs[0:2]
                perplexities.append(math.exp(loss))
                prob = torch.max(softmax(y_pred), dim=1)[0]
                probability.append(prob.cpu().numpy())
        return (perplexities, probability)






    def generate_embeddings(self, inputs, eval_data = None):
        eval_data = paragraph_segmentation(inputs)
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for input_text in eval_data:
                inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
                inputs['labels'] = inputs['input_ids']
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs[3][-1] ### shape = torch.Size([1, 1024, 768]))
                embeddings.append(np.sum(hidden_states.detach().cpu().numpy(), axis=1))
        embeddings = np.concatenate(embeddings)
        return embeddings  ### shape = (7, 768)





    def extract(self, train_dataloader):
        all_embeddings = []
        self.model.eval()
        for step, batch in enumerate(tqdm(train_dataloader)):  ## input shape = torch.Size([1, 1024])
            outputs = self.model(**batch, output_hidden_states=True)
            """outputs = (loss, logits = shape = torch.Size([1, 1024, 50257]), past_key_values, hidden_states shape = tuple (3) -> outputs: torch.Size([1, 1024, 768]))
            """
            embeddings = torch.sum(outputs[3][-1], dim=1)  ### shape = torch.Size([1, 1024, 768]))
            all_embeddings.append(embeddings.detach().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)









    # def __init__(self) -> None:
    #     pass


    # def load_model(self, saved_model_dir = "gpt2"):
    #     """
    #     # Load pretrained model and tokenizer
    #     # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    #     """

    #     config = AutoConfig.from_pretrained(saved_model_dir)
    #     tokenizer = AutoTokenizer.from_pretrained(saved_model_dir, use_fast=True)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         saved_model_dir,
    #         from_tf=bool(".ckpt" in saved_model_dir),
    #         config=config,
    #     ).to(device)

    #     return tokenizer, model
        






    # def generate_embeddings(self, inputs, eval_data = None):

    #     ### convert inputs into fragments
    #     if eval_data is None:
    #         eval_data = paragraph_segmentation(inputs)

    #     self.model.eval()
    #     embeddings = []
    #     with torch.no_grad():
    #         for input_text in eval_data:
    #             inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
    #             inputs['labels'] = inputs['input_ids']
    #             outputs = self.model(**inputs, output_hidden_states=True)
    #             hidden_states = outputs[3][-1] ### shape = torch.Size([1, 1024, 768]))
    #             embeddings.append(hidden_states.detach().cpu().numpy())
    #     embeddings = np.sum(np.hstack(embeddings), axis=1)
    #     return (embeddings)  ### shape = (7, 102, 768)








    # def __init__(self, mode="inference") -> None:

    #     with open(os.path.join("../configs/quality_model_config.json"), 'r') as fp:
    #         self.configs = json.load(fp)
    #     self.saved_model_dir = os.path.join(self.configs["save_dirs"]["root_dir"], self.configs["save_dirs"]["model_dir"])
    #     self.tokenizer, self.model = self.load_model()
    #     if mode == "inference":
    #         print(f"=== Loading Writing Model (Quality Model) from: {self.saved_model_dir}")
    #         self.tokenizer.from_pretrained(self.saved_model_dir)
    #         self.model.load_state_dict(torch.load(os.path.join(self.saved_model_dir, "pytorch_model.bin"),  map_location=device))
    #     self.model.resize_token_embeddings(len(self.tokenizer))



    # def load_model(self):
    #     """
    #     # Load pretrained model and tokenizer
    #     # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    #     """
    #     config = AutoConfig.from_pretrained(self.configs["model_name_or_path"])
    #     tokenizer = AutoTokenizer.from_pretrained(self.configs["model_name_or_path"], use_fast=True)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         self.configs["model_name_or_path"],
    #         from_tf=bool(".ckpt" in self.configs["model_name_or_path"]),
    #         config=config,
    #     ).to(device)

    #     return tokenizer, model
        