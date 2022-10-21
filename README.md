[src_logo_w/o_text]:"assets/logo_wotext.png"
[src_logo_text]: "assets/logo_new.png"


# “Hey ConvXAI<img src="assets/logo_wotext.png" width="32"> , how should I improve?”: Towards Conversational Explainable AI Support for Scientific Writing


This repository is the open source of <b>Conversational XAI (ConvXAI)</b> system to support scientific writing tasks in the human-AI collaborative scenarios.

## How to run the ConvXAI<img src="assets/logo_wotext.png" width="15"> system?

### Installation
```
pip install -r requirements.txt
```

### Prepare Writing Models to be Explained
You can download our pre-trained writing models or train by your own (See how to train writing models from scratch below).
```
download link.
```


### Prepare PyMongo Database 
```
```
Refere to Pymongo for detailed tutorial: `https://pymongo.readthedocs.io/en/stable/`

### Run the ConvXAI system
Open one terminal for running the server. The server is for 
```
bash runners/main_server.sh
```

Open another terminal for running the client.
```
bash runners/main_client.sh
```

### Have fun chatting with ConvXAI<img src="assets/logo_wotext.png" width="15"> robot for improving your paper writing!

<br>

## Understand How ConvXAI<img src="assets/logo_wotext.png" width="15"> System Works
We break down the complicated system into three main modules. Therefore, you can refer to partial or all the modules for your research.

- Module1 - System Services
- Module2 - AI Writing Models
- Module3 - ConvXAI Models


<img src="assets/system_overview.jpeg" width="500">

### Module1 - System Services

We refer to Pailai repository to build up our server-client architecture.


### Module2 - AI Writing Models

We include two AI writing models.





### Module3 - ConvXAI Models

We design our ConvXAI module to include Natural Language Understanding (NLU), AI Explainers (XAIers), Natural Language Generation (NLG) modules.






