
# XAI Intent Detection Task

This **XAI_Intent_Detection** task addresses the problem of classifying each XAI user question into a specific pre-defined XAI intent, which will be used to generate the corresponding AI explanation next. 

This repository includes the code to: 

    - Step1: create a XAI_user_questions dataset for your application;
    - Step2: use this dataset to train a XAI_intent_classifier that can classify the XAI intent for each XAI question.


**Step1:** This step requires three sub-steps to finish:

1. **Create XAI Question Seed**: create a human-labeled `xai_question_seed.csv` that contains N samples for each XAI intent in your application (e.g., we include 11 categories in this system).

2. **Generate XAI Questions**: run the below bash script to generate more XAI questions for each intent category using the GPT model.
```bash
python chatgpt_generate_xai_questions.py --data_folder '../data'
```

OR use the below prompt example (i.e., [data statistics] category) to query [ChatGPT](https://chat.openai.com/) for generating the XAI questions for each category.

```
Here are eleven intents:
 - [important words] What are the key words for this prediction?
 - [other] Could you provide additional context or details to better understand your requirements?
 - [model description] What models were used for this project?
 - [xai tutorial] Can you explain this sentence review?
 - [prediction confidence] How sure is the system about this prediction?
 - [sentence length] What's the statistics of the sentence lengths?
 - [similar examples] Are there any similar examples to this?
 - [label distribution] How are the structure labels distributed?
 - [counterfactual prediction] How can I change the input to alter the outcome?
 - [data statistics] What data for stats?
 - [quality score] What's the range of the style quality scores?


Generate 100 samples for [data statistics] intent. Put the generated samples into a list without index. Please try to vary the text (both length, style and formal or not) and sort them from short to long.


Here are previous samples. Do not repeat.

What data for stats?
Info on data?
Data source?
Source of stats data?
Stats from what data?
What kind of data does the system learn from?
What is the source of the data?
How were the labels/ground-truth produced?
What is the sample size?
```


3. **Human Confirmation**: ask humans to confirm each XAI question and remove the disagreed ones.

4. **Split into Datasets**: split the all XAI questions from `xai_intent_all.csv` into train/val/test datasets by running:
```bash
python process_data_intent.py \
        --input-file ../data/xai_intent_all.csv \
        --output-folder ../data/xai_intent_dataset
```


The **Train/Val/Test** dataset sizes in this system are: 700/100/200. The dataset is saved at: `./data/xai_intent_dataset/`.


| **XAI Types**               | **DataSizze** |
|-----------------------------|---------------|
| [data statistics]           | 100           |
| [model description]         | 100           |
| [quality score]             | 80            |
| [label distribution]        | 80            |
| [sentence length]           | 80            |
| [prediction confidence]     | 100           |
| [similar examples]          | 100           |
| [important words]           | 100           |
| [counterfactual prediction] | 100           |
| [xai tutorial]              | 80            |
| [other]                     | 80            |






**Step2:** Train the XAI intent classification model by running:
```bash
bash ./src/run_scripts.sh
```

We leverage the Deberta model in this system and the performance of **deberta-v3-xsmall** model is:

| **XAI Types**               | **precision** | **recall** | **f1-score** | **support** |
|-----------------------------|---------------|------------|--------------|-------------|
| [data statistics]           | 1.0           | 0.9        | 0.95         | 10          |
| [model description]         | 0.90          | 0.9        | 0.9          | 10          |
| [quality score]             | 1.0           | 1.0        | 1.0          | 8           |
| [label distribution]        | 1.0           | 1.0        | 1.0          | 8           |
| [sentence length]           | 1.0           | 1.0        | 1.0          | 8           |
| [prediction confidence]     | 0.91          | 1.0        | 0.95         | 10          |
| [similar examples]          | 1.0           | 1.0        | 1.0          | 10          |
| [important words]           | 1.0           | 1.0        | 1.0          | 10          |
| [counterfactual prediction] | 1.0           | 1.0        | 1.0          | 10          |
| [xai tutorial]              | 1.0           | 1.0        | 1.0          | 8           |
| [other]                     | 1.0           | 1.0        | 1.0          | 8           |
| **accuracy**                |               |            | **0.98**     | **100**     |
| **macro avg**               | **0.98**      | **0.98**   | **0.98**     | **100**     |
| **weighted avg**            | **0.98**      | **0.98**   | **0.98**     | **100**     |



