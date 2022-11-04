

# ConvXAI<img src="assets/logo_wotext.png" width="38">
This repository includes code for the ConvXAI system as described in the paper:

>[“Hey ConvXAI<img src="assets/logo_wotext.png" width="21"> , how should I improve?”: Towards Conversational Explainable AI Support for Scientific Writing](https://hua-shen.org/assets/files/convxai.pdf)  
> Hua Shen, Chieh-Yang Huang, Tongshuang Wu, Ting-Hao (Kenneth) Huang.
> Arxiv, 2022.



Bibtex for citations:
```bibtex
 @inproceedings{convxai:arxiv:2022,  
 author = {Hua Shen and Chieh-Yang Huang and Tongshuang Wu and Ting-Hao (Kenneth) Huang},  
 title = {“Hey ConvXAI, how should I improve?”: Towards Conversational Explainable AI Support for Scientific Writing},  
 booktitle = {Arxiv},  
 year = {2022}
 }
```



Table of Contents
=================

   * [ConvXAI<img src="assets/logo_wotext.png" width="18">](#convxai)
      * [Table of Contents](#table-of-contents)
      * [Prerequisites](#prerequisites)
         * [Installation](#installation)
         * [MongoDB setup](#mongodb-setup)
         * [Config files setup](#config-files-setup)
         * [Check pretrained data and models](#check-pretrained-data-and-models)
      * [How to run ConvXAI](#how-to-run-convxai)
         * [Run the server](#run-the-server)
         * [Run the client](#run-the-client)
         * [Browse UI to interact](#browse-ui-to-interact)
      * [ConvXAI Tutorials](#convxai-tutorials)
         * [1. ConvXAI Overview](#1-covnxai-overview)
         * [2. UI Web Service](#2-ui-web-service)
         * [3. AI Writing Models](#3-ai-writing-models)
         * [4. Conversational XAI Models](#4-conversational-xai-models)
         * [5. System Infrastructure](#5-system-infrastructure)



## Prerequisites


### Installation
**Create** a `convxai` virtual environment, **activate** the environment, and **install** the libraries for ConvXAI as below.

```bash
$conda create --name convxai python=3.7
$conda activate convxai
$pip install -r requirements.txt
$conda install -c conda-forge git-lfs
```

<!-- TODO
From Pypi:
```bash
pip install convxai
```

From source:
```bash
git clone git@github.com:huashen218/convxai.git
cd convxai
pip install -e .
```
 -->


### MongoDB setup
ConvXAI system is built upon [MongoDB](https://www.mongodb.com/) database. Please install [MongoDB](https://www.mongodb.com/) on your node and ensure you have the database access to connect and manage the data.
Then refer to the [Config Files Setup](#config-files-setup) section to set up [`mongodb_config.yml`](convxai/configs/mongodb_config.yml).

### Config files setup
Set up the  configs files of ConvXAI under path `convxai/configs`:

   * [mongodb_config.yml](convxai/configs/mongodb_config.yml):  You can either deploy both server and client in the **same machine** setting `mongo_host: localhost`, or you can deply them on **two machines** and set your *client machine's IP address* as mongo_host, e.g., `mongo_host: "157.230.188.155""`.

```yaml
mongo_host: localhost
mongo_db_name: convxai
```


   * [configs.yml](convxai/configs/configs.yml)

Set up the path for both **scientific writing models** and the pre-trained checkpoints of **conversational XAI models**.

**Scientific writing models**: ConvXAI involves a SciBERT-based *writing structure model* (i.e., diversity model) and a GPT-based *writing style model* (i.e., quality model). 

The [diversity model](https://huggingface.co/huashen218/convxai-diversity-model?text=I+like+you.+I+love+you) and [quality model](https://huggingface.co/huashen218/convxai-quality-model?text=My+name+is+Merve+and+my+favorite)
are both accessible from the [Huggingface Hub](https://huggingface.co/models) and will be downloaded with below script.

```yaml
scientific_writing:
    diversity_model_dir: "huashen218/convxai-diversity-model"
    quality_model_dir: "huashen218/convxai-quality-model"
```


**Conversational XAI models**: Please specify the `path_of_convxai/` in the `checkpoints_root_dir` shown below. 
For instance, a user clone the convxai repo under `/home/huashen/workspace/projects/` path, then the `path_of_convxai` is `/home/hqs5468/hua/workspace/projects/convxai`. 

```yaml
conversational_xai:
    checkpoints_root_dir: "path_of_convxai/checkpoints/xai_models/"
    xai_example_dir:
        xai_emample_embeddings_dir:
            ACL: "xai_example_embeddings/diversity_model_ACL_embeddings.h5"
            CHI: "xai_example_embeddings/diversity_model_CHI_embeddings.h5"
            ICLR: "xai_example_embeddings/diversity_model_ICLR_embeddings.h5"
        xai_emample_texts_dir:
            ACL: "xai_example_embeddings/diversity_model_ACL_texts.h5"
            CHI: "xai_example_embeddings/diversity_model_CHI_texts.h5"
            ICLR: "xai_example_embeddings/diversity_model_ICLR_texts.h5"
    xai_writing_aspect_prediction_dir: "/xai_writing_aspect_prediction"
    xai_counterfactual_dir: "xai_writing_aspect_prediction/"
```

   * [service_config.yml](convxai/configs/service_config.yml): You can keep this file unchanged unless you want to change the `relative paths` or the `class names` inside of `service_config.yml`.


### Check pretrained data and models
You can **skip this step** if you are going to **use the default datasets and models of ConvXAI system**, because ConvXAI repository is **self-contained**. It includes:
<!-- - **Two AI writing models**: are uploaded to Huggingface Hub. One is a *SciBERT-based classification* model (i.e., `huashen218/convxai-quality-model`), the other is a *GPT-based generative* model (i.e., `huashen218/convxai-quality-model`). The models will be automatically downloaded when deploying ConvXAI. -->
- **CIA dataset**: collects paper abstracts from 2018-2022 in **C**HI, **I**CLR and **A**CL conferences. CIA dataset is for finetuning *GPT-based* model to generate scientific style quality scores. Data path is: `data/CIA`.
- **XAI models**: contains pretrained checkpoints supporting conversational XAI modules to generate AI comments and explanations on-the-fly. Particularly, the `checkpoints/` include:
   * `xai_writing_aspect_prediction/`: enables xai_models to generate AI comments related to the submitted paper's apsect label distribution.
   * `xai_example_embeddings/`: saves embeddings from CIA datasets to enable xai_models to generate example-based explanations. The method is **NN\_DOT** method described in [this paper](https://aclanthology.org/2021.naacl-main.75.pdf).
   * `xai_counterfactual_explainer_models/`: contains [MiCE](https://aclanthology.org/2021.findings-acl.336.pdf) counterfactual model pre-trained on our writing structure model.

You can also train your own writing and XAI models from scratch. Please refer to the [ConvXAI Tutorial](#convxai-tutorials) for details.


## How to Run ConvXAI

You can deploy the ConvXAI **server** (i.e., deep learning server for writing and XAI models) and **client** (i.e., UI web service) either on the *same node* OR on *two different nodes*.

Then please run server and client on two different terminals as described below.


### Run the server:
One terminal runs the server with: `$bash path_of_convxai/convxai/runners/main_server.sh`. 

Please specify the `path_of_convxai/` inside the [main_server.sh](convxai/runners/main_server.sh) shown below. 
You can also change `--port` if needed.
```bash
#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=path_of_convxai/
RUN_SERVICE_DIR="path_of_convxai/convxai";
CUDA_VISIBLE_DEVICES=0 python $RUN_SERVICE_DIR/services/run_server/run.py \
                        --config-path $RUN_SERVICE_DIR/configs/service_config.yml \
                        --port 10020;
```




### Run the client:
The other terminal runs the client with: `$bash path_of_convxai/convxai/runners/main_client.sh`. Please specify the `path_of_convxai/` similarly.

```bash
#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=path_of_convxai/
RUN_SERVICE_DIR="/path_of_convxai/convxai/services/web_service";
python $RUN_SERVICE_DIR/web_server.py
```


### Browse UI to interact:
Then check the client terminal output, such as `-  * Running on http://157.230.188.155:8080/ (Press CTRL+C to quit)`, to open the browser link to interact with ConvXAI user interface.

Have fun chatting with ConvXAI<img src="assets/logo_wotext.png" width="18"> robot for improving your paper writing!



## ConvXAI Tutorials


### 1. CovnXAI Overview

We demonstrate the architecture of ConvXAI in Figure1. ConvXAI mainly includes **four modules** summarized as below: 
- **UI web service** (i.e., in **client**): leverages [Flask](https://flask.palletsprojects.com/en/2.2.x/) to support human-AI interactions; 
- **Deep learning AI writing models** (i.e., in **server**)： generates instance-wise AI predictions;
- **Conversational XAI models** (i.e., in **server**): generate human-understandable AI comments, receive user questions and generate free-text XAI responses; 
- **System Infrastructures** (e.g., [WebSocket](https://en.wikipedia.org/wiki/WebSocket) protocol, [MongoDB](https://www.mongodb.com/) database) to support communication between the client and server.

We further introduce details of each module in the following sections. 

<!-- | ![](assets/github_framework.png) |  -->
| <img src="assets/github_framework.png" width="450">| 
|:--:| 
| **Figure1. Overview of ConvXAI Architecture** |



### 2. UI Web Service

You can check the ConvXAI web service files at `convxai/services/web_service/`. 

Specifically, ConvXAI deploys UI with [Flask](https://flask.palletsprojects.com/en/2.2.x/). It includes:
- `web_server.py` - run Flask and Websocket and MongoDB connection.
- `templates/user_interface.html` - the `html` file to host UI.
- `static` - include all `css` and `javascript` files of the UI. The UI designs of *writing models* and *XAI chatbot* are specified in `writing.css(.js)` and  `chatbot.css(.js)` files, respectively.


### 3. AI Writing Models


Check the AI writing model specifications and data preprocessing files at `convxai/writing_models/models` and `convxai/writing_models/dataloaders`, respectively.

If you want to **train the writing models from scratch**, please check the `convxai/writing_models/trainers/`.

For a brief summary of writing models, ConvXAI includes the pre-trained [diversity model](https://huggingface.co/huashen218/convxai-diversity-model?text=I+like+you.+I+love+you) and [quality model](https://huggingface.co/huashen218/convxai-quality-model?text=My+name+is+Merve+and+my+favorite) AI writing models, which are both accessible from the [Huggingface Hub](https://huggingface.co/models).

- **diversity model**: a SciBERT-based *writing structure model* with five classification categories fine-tuned on the [CODA-19](https://github.com/windx0303/CODA-19) dataset.


- **quality model**: a GPT-based *writing style model* generating quality scores based on the tokens' perplexity values. The model is fine-tuned on the [CIA dataset](#check-pretrained-data-and-models).






### 4. Conversational XAI Models

Please check the conversational XAI models at [convxai/xai_models/](convxai/xai_models/). ConvXAI specifies the `XaiAgent`, `XAIExplainer` and `AICommenter` in the `convxai/xai_models/xaiagent.py` file.

Further, referring to the cutting-edge dialog systems, the ConvXAI explanation process also include **three modules**:
- [Natural Language Understanding (NLU)](convxai/xai_models/models/modules/nlu.py)
- [AI Explainers (XAIers)](convxai/xai_models/models/modules/explainer.py)
- [ Natural Language Generation (NLG)](convxai/xai_models/models/modules/nlg.py)



### 5. System Infrastructure

Check the system infrastructure files at [convxai/services/](convxai/services/). 
You can further check the script to host the server at [convxai/services/run_server/](convxai/services/run_server/) and details of websocket and mongodb specifications at [convxai/services/websocket/](convxai/services/websocket/).

Note that ConvXAI refers to the [ParlAI](https://parl.ai/) platform to build the server-client architecture.
ConvXAI uses [PyMongo](https://pymongo.readthedocs.io/en/stable/) python package to work with MongoDB.















