

## ConvXAI Tutorials


Table of Contents
=================

   * [ConvXAI<img src="../assets/logo_wotext.png" width="18">](#convxai)
      * [ConvXAI Tutorials](#convxai-tutorials)
         * [1. ConvXAI Overview](#1-covnxai-overview)
         * [2. UI Web Service](#2-ui-web-service)
         * [3. AI Writing Models](#3-ai-writing-models)
         * [4. Conversational XAI Models](#4-conversational-xai-models)
         * [5. System Infrastructure](#5-system-infrastructure)


### 1. CovnXAI Overview

We demonstrate the architecture of ConvXAI in Figure1. ConvXAI mainly includes **four modules** summarized as below: 
- **UI web service** (i.e., in **client**): leverages [Flask](https://flask.palletsprojects.com/en/2.2.x/) to support human-AI interactions; 
- **Deep learning AI writing models** (i.e., in **server**)： generates instance-wise AI predictions;
- **Conversational XAI models** (i.e., in **server**): generate human-understandable AI comments, receive user questions and generate free-text XAI responses; 
- **System Infrastructures** (e.g., [WebSocket](https://en.wikipedia.org/wiki/WebSocket) protocol, [MongoDB](https://www.mongodb.com/) database) to support communication between the client and server.

We further introduce details of each module in the following sections. 

<!-- | ![](assets/github_framework.png) |  -->
| <img src="../assets/github_framework.png" width="450">| 
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


