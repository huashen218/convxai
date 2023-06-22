#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import time
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version


from transformers import GPT2Tokenizer, OPTModel, OPTConfig, OPTForCausalLM
# HuggingFace OPTModel API: https://huggingface.co/docs/transformers/model_doc/opt#opt


# Tensorboard with Pytorch: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
# $ tensorboard --logdir=runs



### Add for tensorboard ###
from torch.utils.tensorboard import SummaryWriter


from convxai.writing_models.utils import *
from convxai.writing_models.models import *
from convxai.writing_models.trainers.quality_trainer.trainer import Trainer
from convxai.writing_models.trainers.quality_trainer.data_loader import *


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def inference_ppl(model, eval_dataloader, eval_dataset, accelerator, configs):

    model.eval()
    losses = []
    perplexities = []
    for step, batch in tqdm(enumerate(eval_dataloader)):
        batch = {k: v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(configs["data_configs"]["per_device_eval_batch_size"])))
        perplexity = math.exp(loss)
        perplexities.append(perplexity)

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]

    perplexities = np.array(perplexities)

    return perplexities





def main(args):

    """Load configuration file
    """
    with open(os.path.join(args.config_dir), 'r') as fp:
        configs = json.load(fp)
    model_dir = configs["save_configs"]["output_dir"]
    create_folder([model_dir])

    writer = SummaryWriter(log_dir=os.path.join(model_dir, "logs"))


    """
    Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    """
    accelerator = Accelerator(log_with="all", logging_dir=configs["save_configs"]["output_dir"]) if configs["save_configs"]["with_tracking"] else Accelerator()
    """Make one log on every process with the configuration for debugging."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if configs["train_configs"]["seed"] is not None:
        set_seed(configs["train_configs"]["seed"])


    """Handle the repository creation"""
    if accelerator.is_main_process:
        if configs["save_configs"]["push_to_hub"]:
            if configs["save_configs"]["hub_model_id"] is None:
                repo_name = get_full_repo_name(Path(configs["save_configs"]["output_dir"]).name, token=configs["save_configs"]["hub_token"])
            else:
                repo_name = configs["save_configs"]["hub_model_id"]
            repo = Repository(configs["save_configs"]["output_dir"], clone_from=repo_name)

            with open(os.path.join(configs["save_configs"]["output_dir"], ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif configs["save_configs"]["output_dir"] is not None:
            os.makedirs(configs["save_configs"]["output_dir"], exist_ok=True)
    accelerator.wait_for_everyone()


    """ Load Model """
    quality_model = QualityModel(saved_model_dir=configs["save_configs"]["output_dir"])


    """ Load Dataset """
    train_dataloader, dev_dataloader, test_dataloader, train_dataset, dev_dataset, test_dataset = load_data(configs, quality_model.tokenizer, accelerator)



    train_ppls = inference_ppl(quality_model.model, train_dataloader, train_dataset, accelerator, configs)
    dev_ppls = inference_ppl(quality_model.model, dev_dataloader, dev_dataset, accelerator, configs)
    test_ppls = inference_ppl(quality_model.model, test_dataloader, test_dataset, accelerator, configs)

    print("train_ppls:", np.array(train_ppls).shape)
    print("dev_ppls:", np.array(dev_ppls).shape)
    print("test_ppls:", np.array(test_ppls).shape)
    np.savez(os.path.join(args.save_ppl_dir, "ppl_distribution.npz"), train_ppls=train_ppls, dev_ppls=dev_ppls, test_ppls=test_ppls)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="The configuration file path.",
    )
    parser.add_argument(
        "--save_ppl_dir",
        type=str,
        default=None,
        help="The configuration file path.",
    )
    args = parser.parse_args()
    main(args)
