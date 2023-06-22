DATA_FOLDER="../data"

# # 1 - generate XAI quesitons using ChatGPT
# python chatgpt_generate_xai_questions.py \
#         --data_folder ${DATA_FOLDER}


# # 2 - split the dataset
# python process_data_intent.py \
#         --input-file ${DATA_FOLDER}/xai_intent_all.csv \
#         --output-folder ${DATA_FOLDER}/xai_intent_dataset


# 3 - train the model
MODEL_FOLDER="/data/data/hua/workspace/projects/convxai/checkpoints/intent_model/intent-deberta-v3-xsmall"
MODEL_NAME="microsoft/deberta-v3-xsmall"
CUDA_VISIBLE_DEVICES=0 python run_glue.py \
  --model_name_or_path ${MODEL_FOLDER} \
  --train_file "${DATA_FOLDER}/xai_intent_dataset/train.json" \
  --validation_file "${DATA_FOLDER}/xai_intent_dataset/valid.json" \
  --test_file "${DATA_FOLDER}/xai_intent_dataset/test.json" \
  --do_eval \
  --do_predict \
  --pad_to_max_length false \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --auto_find_batch_size \
  --learning_rate 1e-5 \
  --num_train_epochs 30 \
  --warmup_ratio 0.05 \
  --fp16 true \
  --output_dir "${MODEL_FOLDER}" \
  --logging_steps 10 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --load_best_model_at_end \
  --metric_for_best_model "macro avg/f1-score"