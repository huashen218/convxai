---
tags:
- generated_from_trainer
model-index:
- name: intent-deberta-v3-xsmall
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# intent-deberta-v3-xsmall

This model is a fine-tuned version of [/data/data/hua/workspace/projects/convxai/checkpoints/intent_model/intent-deberta-v3-xsmall](https://huggingface.co//data/data/hua/workspace/projects/convxai/checkpoints/intent_model/intent-deberta-v3-xsmall) on an unknown dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.9740
- eval_macro-precision: 0.9826
- eval_micro-precision: 0.98
- eval_macro-recall: 0.9818
- eval_micro-recall: 0.98
- eval_macro-f1: 0.9818
- eval_micro-f1: 0.98
- eval_[counterfactual prediction]/precision: 1.0
- eval_[counterfactual prediction]/recall: 1.0
- eval_[counterfactual prediction]/f1-score: 1.0
- eval_[counterfactual prediction]/support: 10
- eval_[data statistics]/precision: 1.0
- eval_[data statistics]/recall: 0.9
- eval_[data statistics]/f1-score: 0.9474
- eval_[data statistics]/support: 10
- eval_[important words]/precision: 1.0
- eval_[important words]/recall: 1.0
- eval_[important words]/f1-score: 1.0
- eval_[important words]/support: 10
- eval_[label distribution]/precision: 1.0
- eval_[label distribution]/recall: 1.0
- eval_[label distribution]/f1-score: 1.0
- eval_[label distribution]/support: 8
- eval_[model description]/precision: 0.9
- eval_[model description]/recall: 0.9
- eval_[model description]/f1-score: 0.9
- eval_[model description]/support: 10
- eval_[other]/precision: 1.0
- eval_[other]/recall: 1.0
- eval_[other]/f1-score: 1.0
- eval_[other]/support: 8
- eval_[prediction confidence]/precision: 0.9091
- eval_[prediction confidence]/recall: 1.0
- eval_[prediction confidence]/f1-score: 0.9524
- eval_[prediction confidence]/support: 10
- eval_[quality score]/precision: 1.0
- eval_[quality score]/recall: 1.0
- eval_[quality score]/f1-score: 1.0
- eval_[quality score]/support: 8
- eval_[sentence length]/precision: 1.0
- eval_[sentence length]/recall: 1.0
- eval_[sentence length]/f1-score: 1.0
- eval_[sentence length]/support: 8
- eval_[similar examples]/precision: 1.0
- eval_[similar examples]/recall: 1.0
- eval_[similar examples]/f1-score: 1.0
- eval_[similar examples]/support: 10
- eval_[xai tutorial]/precision: 1.0
- eval_[xai tutorial]/recall: 1.0
- eval_[xai tutorial]/f1-score: 1.0
- eval_[xai tutorial]/support: 8
- eval_accuracy: 0.98
- eval_macro avg/precision: 0.9826
- eval_macro avg/recall: 0.9818
- eval_macro avg/f1-score: 0.9818
- eval_macro avg/support: 100
- eval_weighted avg/precision: 0.9809
- eval_weighted avg/recall: 0.98
- eval_weighted avg/f1-score: 0.9800
- eval_weighted avg/support: 100
- eval_runtime: 0.4377
- eval_samples_per_second: 228.481
- eval_steps_per_second: 29.703
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 30.0
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.23.0
- Pytorch 1.10.2+cu102
- Datasets 2.12.0
- Tokenizers 0.13.3
