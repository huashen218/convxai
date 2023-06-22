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
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
)
# from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version

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
            
            with open(os.path.join(configs["save_configs"]["output_dir"], ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif configs["save_configs"]["output_dir"] is not None:
            os.makedirs(configs["save_configs"]["output_dir"], exist_ok=True)
    accelerator.wait_for_everyone()


    """ Load Model """
    quality_model = QualityModel()


    """ Load Dataset """
    train_dataloader, dev_dataloader, test_dataloader, train_dataset, dev_dataset, test_dataset = load_data(configs, quality_model.tokenizer, accelerator)



    """ Training Settings """

    ### Optimizer:  Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in quality_model.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": configs["train_configs"]["weight_decay"],
        },
        {
            "params": [p for n, p in quality_model.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs["train_configs"]["learning_rate"])

    ### On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        quality_model.model.tie_weights()

    ### Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / configs["train_configs"]["gradient_accumulation_steps"])
    if configs["train_configs"]["max_train_steps"] is None:
        configs["train_configs"]["max_train_steps"] = configs["train_configs"]["num_train_epochs"] * num_update_steps_per_epoch
    else:
        configs["train_configs"]["num_train_epochs"] = math.ceil(configs["train_configs"]["max_train_steps"] / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=configs["train_configs"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=configs["train_configs"]["num_warmup_steps"],
        num_training_steps=configs["train_configs"]["max_train_steps"],
    )

    ### Prepare everything with our `accelerator`.
    quality_model.model, optimizer, train_dataloader, dev_dataloader, lr_scheduler = accelerator.prepare(
        quality_model.model, optimizer, train_dataloader, dev_dataloader, lr_scheduler
    )

    ### We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / configs["train_configs"]["gradient_accumulation_steps"])
    configs["train_configs"]["max_train_steps"] = configs["train_configs"]["num_train_epochs"] * num_update_steps_per_epoch


    ### We need to initialize the trackers we use, and also store our configuration
    if configs["save_configs"]["with_tracking"]:
        # TensorBoard cannot log Enums, need the raw value
        # experiment_config["lr_scheduler_type"] = configs["train_configs"]["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", configs["train_configs"]["lr_scheduler_type"].value)

    ### Train!
    total_batch_size = configs["data_configs"]["per_device_train_batch_size"] * accelerator.num_processes * configs["train_configs"]["gradient_accumulation_steps"]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {configs['train_configs']['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {configs['data_configs']['per_device_train_batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {configs['train_configs']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {configs['train_configs']['max_train_steps']}")

    completed_steps = 0
    starting_epoch = 0
    resume_step = None


    ### Potentially load in the weights and states from a previous save
    if configs["save_configs"]["resume_from_checkpoint"]:
        if configs["save_configs"]["resume_from_checkpoint"] is not None or configs["save_configs"]["resume_from_checkpoint"] != "":
            accelerator.print(f"Resumed from checkpoint: {configs['save_configs']['resume_from_checkpoint']}")
            accelerator.load_state(configs['save_configs']['resume_from_checkpoint'])
            path = os.path.basename(configs['save_configs']['resume_from_checkpoint'])
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)


    trainer = Trainer(configs, quality_model.model, quality_model.tokenizer, accelerator, writer, optimizer, lr_scheduler)

    print('\t-----------------------------------------------------------------------')
    print(f'\t ============= Model Training with Training and Validation Sets =============')
    print('\t-----------------------------------------------------------------------')
    print(f"=== starting_epoch: {starting_epoch}, num_train_epochs = {configs['train_configs']['num_train_epochs']} ===")
    start_time = time.time()
    """Train with Training set and Validation set"""
    for epoch in range(starting_epoch, configs["train_configs"]["num_train_epochs"]):
        print(f'\t ========================== Epoch: {epoch:02} ==========================')
        trainer.train(epoch, train_dataloader, starting_epoch, resume_step)
        perplexity = trainer.evaluate(epoch, dev_dataloader, dev_dataset, tf_writer="validation")
    end_time = time.time()
    print("Epoch{}: - Total Time: {} sec".format(configs["train_configs"]["num_train_epochs"], str(end_time-start_time)))



    print('\t-----------------------------------------------------------------------')
    print(f'\t ============= Model Evaluating on Test Set =============')
    print('\t-----------------------------------------------------------------------')
    quality_model = QualityModel(saved_model_dir=configs["save_configs"]["output_dir"])


    trainer = Trainer(configs, quality_model.model, quality_model.tokenizer, accelerator, writer, optimizer, lr_scheduler)
    perplexity = trainer.evaluate(configs["train_configs"]["num_train_epochs"], test_dataloader, test_dataset, tf_writer = "test")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="The configuration file path.",
    )
    args = parser.parse_args()
    main(args)



