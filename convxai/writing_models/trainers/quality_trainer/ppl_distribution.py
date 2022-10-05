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






# #!/usr/bin/env python
# # coding=utf-8
# # Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
# on a text file or a dataset without using HuggingFace Trainer.
# Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
# https://huggingface.co/models?filter=text-generation

# code reference: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
# """
# # You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.


# import argparse
# import logging
# import math
# import os
# import random
# from itertools import chain
# from pathlib import Path

# import datasets
# import torch
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm

# import transformers
# from accelerate import Accelerator, DistributedType
# from huggingface_hub import Repository
# from transformers import (
#     CONFIG_MAPPING,
#     MODEL_MAPPING,
#     AdamW,
#     AutoConfig,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     SchedulerType,
#     default_data_collator,
#     get_scheduler,
#     set_seed,
# )
# from transformers.utils.versions import require_version


# from models import *
# from data_loader import *
# from trainer import Trainer


# logger = logging.getLogger(__name__)

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

# MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)






# def inference_ppl(model, eval_dataloader, eval_dataset, accelerator, args):

#     model.eval()
#     losses = []
#     perplexities = []
#     for step, batch in tqdm(enumerate(eval_dataloader)):
#         batch = {k: v.to(device) for k,v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)
#         loss = outputs.loss
#         # loss_batch = accelerator.gather(loss.repeat(args.per_device_batch_size))
#         losses.append(accelerator.gather(loss.repeat(args.per_device_batch_size)))
#         perplexity = math.exp(loss)
#         perplexities.append(perplexity)

#     losses = torch.cat(losses)
#     losses = losses[: len(eval_dataset)]

#     perplexities = np.array(perplexities)

#     return perplexities





# def main(args):

#     ############# Set Configs ############
#     with open(os.path.join(args.config_dir), 'r') as fp:
#         configs = json.load(fp)
#     model_dir = os.path.join(configs["save_dirs"]["root_dir"], configs["save_dirs"]["model_dir"])
#     create_folder([model_dir])


#     if args.push_to_hub:
#         assert model_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."


#     ############# Set Accelerator ###########c#
#     """
#     # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
#     # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
#     """
#     accelerator = Accelerator(log_with="all", logging_dir=model_dir) if args.with_tracking else Accelerator()
#     # Make one log on every process with the configuration for debugging.
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
#     )
#     logger.info(accelerator.state)   ## , main_process_only=False

#     if accelerator.is_local_main_process:
#         datasets.utils.logging.set_verbosity_warning()
#         transformers.utils.logging.set_verbosity_info()
#     else:
#         datasets.utils.logging.set_verbosity_error()
#         transformers.utils.logging.set_verbosity_error()

#     # If passed along, set the training seed now.
#     if args.seed is not None:
#         set_seed(args.seed)


#     # Handle the repository creation
#     if accelerator.is_main_process:
        
#         if args.push_to_hub:
#             if args.hub_model_id is None:
#                 repo_name = get_full_repo_name(Path(model_dir).name, token=args.hub_token)
#             else:
#                 repo_name = args.hub_model_id
#             repo = Repository(model_dir, clone_from=repo_name)

#             with open(os.path.join(model_dir, ".gitignore"), "w+") as gitignore:
#                 if "step_*" not in gitignore:
#                     gitignore.write("step_*\n")
#                 if "epoch_*" not in gitignore:
#                     gitignore.write("epoch_*\n")
#         elif model_dir is not None:
#             os.makedirs(model_dir, exist_ok=True)

#     accelerator.wait_for_everyone()



#     ############# Load Model ############
#     quality_model = QualityModel()
#     tokenizer, model = quality_model.load_model(model_dir)


#     ############ Load Dataset ############
#     # train_dataloader, dev_dataloader, train_dataset, dev_dataset = load_data(args, configs, tokenizer, accelerator)
#     train_dataloader, dev_dataloader, test_dataloader, train_dataset, dev_dataset, test_dataset = load_data(args, configs, tokenizer, accelerator)


#     train_ppls = inference_ppl(model, train_dataloader, train_dataset, accelerator, args)
#     dev_ppls = inference_ppl(model, dev_dataloader, dev_dataset, accelerator, args)
#     test_ppls = inference_ppl(model, test_dataloader, test_dataset, accelerator, args)

#     print("train_ppls:", np.array(train_ppls).shape)
#     print("dev_ppls:", np.array(dev_ppls).shape)
#     print("test_ppls:", np.array(test_ppls).shape)
#     np.savez(os.path.join(args.save_ppl_dir, "ppl_distribution.npz"), train_ppls=train_ppls, dev_ppls=dev_ppls, test_ppls=test_ppls)






# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
   
#     ###### configs file ######
#     parser.add_argument(
#         "--config_dir",
#         type=str,
#         default=None,
#         help="The configuration file path.",
#     )

#     parser.add_argument("--mode", dest="mode", help="train/evaluate", type=str, default="train")

#     parser.add_argument(
#         "--save_ppl_dir",
#         type=str,
#         default=None,
#         help="The configuration file path.",
#     )


#     ###### dataset configs ######
#     parser.add_argument(
#         "--dataset_name",
#         type=str,
#         default=None,
#         help="The name of the dataset to use (via the datasets library).",
#     )
#     parser.add_argument(
#         "--dataset_config_name",
#         type=str,
#         default=None,
#         help="The configuration name of the dataset to use (via the datasets library).",
#     )
#     parser.add_argument(
#         "--validation_split_percentage",
#         default=5,
#         help="The percentage of the train set used as validation set in case there's no validation split",
#     )
#     parser.add_argument(
#         "--block_size",
#         type=int,
#         default=None,
#         help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
#     )
#     parser.add_argument(
#         "--no_keep_linebreaks", 
#         action="store_true", 
#         help="Do not keep line breaks when using TXT files."
#     )
#     parser.add_argument(
#         "--overwrite_cache", 
#         type=bool, 
#         default=False, 
#         help="Overwrite the cached training and evaluation sets"
#     )


#     ###### model configs ######
#     parser.add_argument(
#         "--config_name",
#         type=str,
#         default=None,
#         help="Pretrained config name or path if not the same as model_name",
#     )
#     parser.add_argument(
#         "--tokenizer_name",
#         type=str,
#         default=None,
#         help="Pretrained tokenizer name or path if not the same as model_name",
#     )
#     parser.add_argument(
#         "--use_slow_tokenizer",
#         action="store_true",
#         help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
#     )


#     ###### model save dir ######
#     parser.add_argument(
#         "--model_type",
#         type=str,
#         default=None,
#         help="Model type to use if training from scratch.",
#         choices=MODEL_TYPES,
#     )
#     parser.add_argument(
#         "--push_to_hub", 
#         action="store_true", 
#         help="Whether or not to push the model to the Hub."
#     )
#     parser.add_argument(
#         "--hub_model_id", 
#         type=str, 
#         help="The name of the repository to keep in sync with the local `output_dir`."
#     )
#     parser.add_argument(
#         "--hub_token", 
#         type=str, 
#         help="The token to use to push to the Model Hub."
#     )


#     ###### training configs ######
#     parser.add_argument(
#         "--num_train_epochs", 
#         type=int, 
#         default=5, 
#         help="Total number of training epochs to perform."
#     )
#     parser.add_argument(
#         "--per_device_batch_size",
#         type=int,
#         default=1,
#         help="Batch size (per device) for the dataloader.",
#     )
#     parser.add_argument(
#         "--learning_rate",
#         type=float,
#         default=5e-5,
#         help="Initial learning rate (after the potential warmup period) to use.",
#     )

#     parser.add_argument(
#         "--seed", 
#         type=int, 
#         default=None, 
#         help="A seed for reproducible training."
#     )
#     parser.add_argument(
#         "--weight_decay", 
#         type=float, 
#         default=0.0, 
#         help="Weight decay to use."
#     )
#     parser.add_argument(
#         "--max_train_steps",
#         type=int,
#         default=None,
#         help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
#     )
#     parser.add_argument(
#         "--gradient_accumulation_steps",
#         type=int,
#         default=1,
#         help="Number of updates steps to accumulate before performing a backward/update pass.",
#     )
#     parser.add_argument(
#         "--lr_scheduler_type",
#         type=SchedulerType,
#         default="linear",
#         help="The scheduler type to use.",
#         choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
#     )
#     parser.add_argument(
#         "--num_warmup_steps", 
#         type=int, 
#         default=0, 
#         help="Number of steps for the warmup in the lr scheduler."
#     )
#     parser.add_argument(
#         "--preprocessing_num_workers",
#         type=int,
#         default=None,
#         help="The number of processes to use for the preprocessing.",
#     )
#     parser.add_argument(
#         "--checkpointing_steps",
#         type=str,
#         default=None,
#         help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
#     )
#     parser.add_argument(
#         "--resume_from_checkpoint",
#         type=str,
#         default=None,
#         help="If the training should continue from a checkpoint folder.",
#     )
#     parser.add_argument(
#         "--with_tracking",
#         action="store_true",
#         help="Whether to load in all available experiment trackers from the environment and use them for logging.",
#     )

#     args = parser.parse_args()

#     main(args)



